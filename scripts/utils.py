import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets
from config import TRANSFORM, BATCH_SIZE, TEST_SIZE, RANDOM_STATE

def load_datasets():
    """
    Loads the MNIST dataset and splits the training set into training and validation subsets.

    Returns:
        train_subset (Subset): Training subset of MNIST.
        val_subset (Subset): Validation subset of MNIST.
        test_dataset (Dataset): Test set of MNIST.
    """
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=TRANSFORM)
    test_dataset = datasets.MNIST('./data', train=False, transform=TRANSFORM)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=train_dataset.targets
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    return train_subset, val_subset, test_dataset

def create_dataloaders(train_subset, val_subset, test_dataset):
    """
    Creates PyTorch DataLoaders for training, validation, and testing datasets.

    Args:
        train_subset (Subset): Training data.
        val_subset (Subset): Validation data.
        test_dataset (Dataset): Test data.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader

def create_distill_loaders(train_subset, val_subset, train_teacher_logits, val_teacher_logits):
    """
    Creates DataLoaders for the knowledge distillation process using teacher logits.

    Args:
        train_subset (Subset): Training data subset.
        val_subset (Subset): Validation data subset.
        train_teacher_logits (Tensor): Teacher model's output logits for training data.
        val_teacher_logits (Tensor): Teacher model's output logits for validation data.

    Returns:
        tuple: DataLoaders for distillation training and validation.
    """
    from distillation import DistillDataset
    
    train_distill_dataset = DistillDataset(train_subset, train_teacher_logits)
    val_distill_dataset = DistillDataset(val_subset, val_teacher_logits)
    
    train_distill_loader = DataLoader(train_distill_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_distill_loader = DataLoader(val_distill_dataset, batch_size=BATCH_SIZE)
    
    return train_distill_loader, val_distill_loader
