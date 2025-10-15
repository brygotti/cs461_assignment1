import argparse
from pathlib import Path
import importlib.util

import torch
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_model
import numpy as np
from torchsummary import summary

from utils import ImageDatasetNPZ, default_transform
from utils import extract_features_and_labels, run_knn_probe, run_linear_probe
from utils import seed_all


def evaluate_model(model, train_dataloader, val_dataloader, knn_k=5):
    """
    Evaluate the model using k-NN probing.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        train_dataset (ImageDatasetNPZ): The training dataset.
        val_dataset (ImageDatasetNPZ): The validation dataset.

    Returns:
        float: k-NN accuracy on the validation set.
    """
    # Extract features and labels
    train_features, train_labels = extract_features_and_labels(model, train_dataloader)
    val_features, val_labels = extract_features_and_labels(model, val_dataloader)

    # Run k-NN probe
    knn_accuracy = run_knn_probe(train_features, train_labels, val_features, val_labels)
    probe_accuracy = run_linear_probe(train_features, train_labels, val_features, val_labels)

    return knn_accuracy, probe_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission-folder", '-i',
        type=str,
        default="$HOME/cs461_assignment1_submission",
        help="Path to the folder containing the model and other required files.",
    )
    parser.add_argument(
        "--data-dir", '-d',
        type=str,
        default="/shared/CS461/cs461_assignment1_data/",
        help="Directory containing the dataset files.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.npz",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="val.npz",
        help="Path to the validation data file.",
    )
    parser.add_argument(
        "--ood-file",
        type=str,
        default="ood.npz",
        help="Path to the OOD data file.",
    )
    args = parser.parse_args()

    if args.submission_folder.startswith("$HOME"):
        args.submission_folder = args.submission_folder.replace("$HOME", str(Path.home()))

    work_dir = Path(args.submission_folder)
    assert work_dir.exists() and work_dir.is_dir(), \
        f"Submission folder {work_dir} does not exist or is not a directory."
            
    files = list(work_dir.iterdir()) 
    
    assert any(f.name == "models.py" for f in files), \
        "`models.py` not found in the submission folder."
    assert any(f.name == "final_model.safetensors" for f in files), \
        "`final_model.safetensors` not found in the submission folder."
    assert any(f.name == "CS461_Assignment1.ipynb" for f in files), \
        "`CS461_Assignment1.ipynb` not found in the submission folder."
    assert any(f.name == "report.md" for f in files), \
        "`report.md` not found in the submission folder."
    # assert set(files) == {
    #     work_dir / "models.py", work_dir / "final_model.safetensors",
    #     work_dir / "CS461_Assignment1.ipynb", work_dir / "report.md"
    # }, \
    #     "Submission folder contains unexpected files."

    spec = importlib.util.spec_from_file_location("custom_model", str(Path(work_dir) / "models.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    seed_all(42)

    ImageEncoder = module.ImageEncoder  
    
    checkpoint_path = work_dir / "final_model.safetensors"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ImageEncoder().to(device)
    print(summary(model, input_size=(3, 64, 64)))
    load_model(model, checkpoint_path)
    model.eval()
    
    try:
        x = torch.randn(1, 3, 64, 64, device=device)
        with torch.no_grad():
            out = model.get_features(x)
        assert out.shape == (1, 1000), \
            f"Encoded features shape is {out.shape}, expected (1, 1000)."
    except Exception as e:
        raise RuntimeError(f"Error during model forward pass: {e}")
    
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset = ImageDatasetNPZ(
        data_dir / args.train_file, transform=default_transform
    )
    val_dataset = ImageDatasetNPZ(
        data_dir / args.val_file, transform=default_transform
    )
    ood_dataset = ImageDatasetNPZ(
        data_dir / args.ood_file, transform=default_transform
    )
    rng = np.random.RandomState(42)
    ood_val_ratio = 0.2
    ood_train_mask = rng.permutation(len(ood_dataset)) >= int(len(ood_dataset) * ood_val_ratio)
    ood_train_dataset = torch.utils.data.Subset(ood_dataset, np.where(ood_train_mask)[0])
    ood_val_dataset = torch.utils.data.Subset(ood_dataset, np.where(~ood_train_mask)[0])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    ood_train_dataloader = DataLoader(ood_train_dataset, batch_size=64, shuffle=True, num_workers=4)
    ood_val_dataloader = DataLoader(ood_val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Evaluate model
    knn_accuracy, probe_accuracy = evaluate_model(model, train_dataloader, val_dataloader)
    print(f"k-NN Accuracy: {knn_accuracy:.2f}%")
    print(f"Linear Probe Accuracy: {probe_accuracy:.2f}%")
    ood_knn_accuracy, ood_probe_accuracy = evaluate_model(model, ood_train_dataloader, ood_val_dataloader)
    print(f"OOD k-NN Accuracy: {ood_knn_accuracy:.2f}%")
    print(f"OOD Linear Probe Accuracy: {ood_probe_accuracy:.2f}%")
    
    print("Evaluation completed.")