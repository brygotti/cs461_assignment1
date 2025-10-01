
from typing import Union
from pathlib import Path
import random

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset


try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    print("sklearnex not installed, using standard sklearn")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




default_transform = T.Compose([
    T.ToTensor(),
])


class ImageDatasetNPZ(Dataset):
    def __init__(self, data_path: Union[str, Path], transform=default_transform):
        self.load_from_npz(data_path)
        self.transform = transform

    def load_from_npz(self, data_path: Union[str, Path]):
        data = np.load(data_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
def extract_features_and_labels(model, dataloader, normalize=False):
    """
    Extract features and labels from a dataloader using the given model.
    model: an encoder model taking as input a batch of images (batch_size, channels, height, width) and outputing either a batch of feature vectors (batch_size, feature_dim) or a list/tuple in which the first element is the batch of feature vectors (batch_size, feature_dim)
    dataloader: a PyTorch dataloader providing batches of (images, labels)
    returns: features (num_samples, feature_dim), labels (num_samples,)
    """
    features = []
    labels = []

    device = next(model.parameters()).device

    for batch in tqdm(dataloader, disable=True):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            feats = model.get_features(x)
        features.append(feats.cpu())
        labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    if normalize:
        features = F.normalize(features, dim=1)

    return features, labels



def run_knn_probe(train_features, train_labels, test_features, test_labels):
    """
    Runs a k-NN probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_features, train_labels)
    test_preds = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy

def run_linear_probe(train_features, train_labels, test_features, test_labels):
    """
    Runs a linear probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logreg.fit(train_features, train_labels)
    test_preds = logreg.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # The following lines are commented out to allow for non-deterministic behavior which can improve performance on some models.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False