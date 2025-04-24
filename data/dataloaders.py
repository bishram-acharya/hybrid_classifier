"""
DataLoader implementations for the hybrid classification model.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ..config import BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, SEED

def get_transforms(train=True):
    """
    Get image transformations for training or testing.
    
    Args:
        train (bool): Whether to include data augmentation for training
        
    Returns:
        torchvision.transforms.Compose: The transformation pipeline
    """
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if train:
        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)

def get_dataloaders(train_dir, test_dir):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Create datasets with transforms
    train_transforms = get_transforms(train=True)
    test_transforms = get_transforms(train=False)
    
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    # Split into train and validation sets
    train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Update validation transform to use test transforms (no augmentation)
    val_dataset.dataset.transform = test_transforms
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader, test_loader, len(full_train_dataset.classes), full_train_dataset.classes
