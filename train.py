from pathlib import Path
import shutil

import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from safetensors.torch import save_model

import wandb
from torchsummary import summary

import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SCIPER = 325403  # Replace with your SCIPER number
LAST_NAME = "Gotti"  # Replace with your last name
FIRST_NAME = "Bryan"  # Replace with your first name

from utils import ImageDatasetNPZ, default_transform, seed_all
from utils import run_knn_probe, run_linear_probe, extract_features_and_labels

from models import ImageEncoder

batch_size = 256
num_workers = 4
pin_memory = True

def collate_fn(batch):
    xs1, xs2, ys = [], [], []
    for (x1, x2), y in batch:
        xs1.append(x1)
        xs2.append(x2)
        ys.append(y)
    return torch.stack(xs1), torch.stack(xs2), torch.tensor(ys)

class SimCLRTransform:

    def __init__(self, size=32, s=0.5, blur_p=0.5):
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        k = 3 if size <= 32 else 5
        base = [
            T.ToPILImage(),
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p),
            T.ToTensor()
        ]
        self.train_transform = T.Compose(base)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
    
def custom_loss_function(z1, z2, tau=0.5):
    """
    Computes NT-Xent loss.
    z1: (batch_size, feature_dim) tensor of normalized projection vectors
    z2: (batch_size, feature_dim) tensor of normalized projection vectors
    returns: loss (scalar)
    """
    B, d = z1.shape
    z = torch.cat([z1, z2], dim=0)              # (2B, d)
    sim = (z @ z.t()) / tau                     # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)

def training_step(*args, **kwargs):
    model, train_loaders, optimizer, custom_loss_function = args
    avg_loss = 0.
    for tl in train_loaders:
        for x1, x2, y in tqdm(tl):
            x1, x2 = x1.to(device), x2.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                proj1 = model(x1)
                proj2 = model(x2)
            loss = custom_loss_function(proj1, proj2)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            avg_loss += loss.item()
    return { "loss": avg_loss / sum([len(tl) for tl in train_loaders]) }

def evaluation_step(*args, **kwargs):
    model, train_loader_no_augment, val_loader_no_augment, oods_train_loader, oods_val_loader = args
    
    train_features, train_labels = extract_features_and_labels(model, train_loader_no_augment)
    test_features, test_labels = extract_features_and_labels(model, val_loader_no_augment)
    oods_train_features, oods_train_labels = extract_features_and_labels(model, oods_train_loader)
    oods_test_features, oods_test_labels = extract_features_and_labels(model, oods_val_loader)
    
    knn_accuracy = run_knn_probe(train_features, train_labels, test_features, test_labels)
    linear_accuracy = run_linear_probe(train_features, train_labels, test_features, test_labels)
    oods_knn_accuracy = run_knn_probe(oods_train_features, oods_train_labels, oods_test_features, oods_test_labels)
    oods_linear_accuracy = run_linear_probe(oods_train_features, oods_train_labels, oods_test_features, oods_test_labels)

    return { "knn_accuracy": knn_accuracy, "linear_accuracy": linear_accuracy, "oods_knn_accuracy": oods_knn_accuracy, "oods_linear_accuracy": oods_linear_accuracy }

total_epochs = 50  # Adjust the number of epochs as needed
warmup_epochs = 10

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / float(warmup_epochs)
    t = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    return 0.0 + 0.5 * (1 - 0.0) * (1 + math.cos(math.pi * t))

# Feel free to adapt and add more arguments
# lr = 1e-3
# weight_decay = 5e-2
# lr_step_size = 10
# lr_gamma = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

if __name__ == "__main__":
    
    seed_all(42)  # For reproducibility, you can use any integer here

    data_dir = Path('./cs461_assignment1_data/')
    checkpoints_dir = Path('checkpoints')
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=False)
    final_model_path = checkpoints_dir / 'model_final.safetensors'

    simclr_transform = SimCLRTransform(size=64)

    train_datasets = [ImageDatasetNPZ(data_dir / 'train.npz', transform=simclr_transform) for i in range(4)]
    train_dataset_no_augment = ImageDatasetNPZ(data_dir / 'train.npz', transform=default_transform)
    val_dataset_no_augment = ImageDatasetNPZ(data_dir / 'val.npz', transform=default_transform)

    rng = np.random.RandomState(42)
    ds_ood = ImageDatasetNPZ(data_dir / 'ood.npz', transform=default_transform)
    ood_val_ratio = 0.2
    train_mask = rng.permutation(len(ds_ood)) >= int(len(ds_ood) * ood_val_ratio)
    ds_oods_train = torch.utils.data.Subset(ds_ood, np.where(train_mask)[0])
    ds_oods_val = torch.utils.data.Subset(ds_ood, np.where(~train_mask)[0])

    train_loaders = [DataLoader(td, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, collate_fn=collate_fn) for td in train_datasets]
    train_loader_no_augment = DataLoader(train_dataset_no_augment, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader_no_augment  = DataLoader(val_dataset_no_augment,  batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    oods_train_loader = DataLoader(ds_oods_train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    oods_val_loader = DataLoader(ds_oods_val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    model = ImageEncoder().to(device)
    summary(model, input_size=(3, 64, 64))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.6, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_epochs = total_epochs
    eval_interval = 3  # Evaluate the model every 'eval_interval' epochs
    save_interval = 10  # Save the model every 'save_interval' epochs

    with wandb.init(project="cs461_assignment1") as run:

        all_train_stats = []
        all_val_stats = []

        for epoch in range(n_epochs):
            # TODO: Implement the training and evaluation loop
            model.train()
            train_stats = training_step(model, train_loaders, optimizer, custom_loss_function)
            all_train_stats.append(train_stats)

            log_dict = {**train_stats, 'epoch': epoch + 1}

            lr_scheduler.step()

            if (epoch + 1) % eval_interval == 0:
                model.eval()
                val_stats = evaluation_step(model, train_loader_no_augment, val_loader_no_augment, oods_train_loader, oods_val_loader)
                all_val_stats.append(val_stats)
                log_dict.update(val_stats)
                print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_stats['loss']:.4f}, Val kNN-5 Accuracy: {val_stats['knn_accuracy']:.2f}, Val Linear Accuracy: {val_stats['linear_accuracy']:.2f}, OOD kNN-5 Accuracy: {val_stats['oods_knn_accuracy']:.2f}, OOD Linear Accuracy: {val_stats['oods_linear_accuracy']:.2f}")

            if (epoch + 1) % save_interval == 0:
                checkpoint_path = checkpoints_dir / f'model_epoch_{epoch+1}.safetensors'
                save_model(model, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

            wandb.log(log_dict)


        # Save the final model
        save_model(model, final_model_path)

    fig, axes = plt.subplots(1, 5, figsize=(12, 5))
    axes[0].plot(range(1, total_epochs+1), [s["loss"] for s in all_train_stats])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average NT-Xent Loss')

    axes[1].plot(range(1, total_epochs+1), [s["knn_accuracy"] for s in all_val_stats])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('k-NN Accuracy')
    axes[1].set_ylim(0, 1)

    axes[2].plot(range(1, total_epochs+1), [s["linear_accuracy"] for s in all_val_stats])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Linear Accuracy')
    axes[2].set_ylim(0, 1)

    axes[3].plot(range(1, total_epochs+1), [s["oods_knn_accuracy"] for s in all_val_stats])
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('OOD k-NN Accuracy')
    axes[3].set_ylim(0, 1)

    axes[4].plot(range(1, total_epochs+1), [s["oods_linear_accuracy"] for s in all_val_stats])
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('OOD Linear Accuracy')
    axes[4].set_ylim(0, 1)

    plt.show()