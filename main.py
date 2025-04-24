"""
Main entry point for the hybrid classifier.
"""
import torch
import numpy as np
from .config import TRAIN_DIR, TEST_DIR, DEVICE, SEED
from .data.dataloaders import get_dataloaders
from .models.hybridnet import HybridNet, get_vit_processor
from .utils.lbp import extract_lbp_features
from .train import train_model
from .evaluate import evaluate_model
from .utils.visualization import plot_training_history


def main(backbone_model=None):
    """
    Main function to run the hybrid classifier.
    
    Args:
        backbone_model (str, optional): Override the backbone model choice.
                                        Options: 'efficientnet', 'resnet'
    """
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Override backbone model if specified
    if backbone_model:
        config.BACKBONE_MODEL = backbone_model
    
    print(f"Using device: {DEVICE}")
    print(f"Using backbone model: {config.BACKBONE_MODEL}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        TRAIN_DIR, TEST_DIR
    )
    
    # Initialize model and processor
    model = HybridNet(num_classes=num_classes, backbone=config.BACKBONE_MODEL).to(DEVICE)
    vit_processor = get_vit_processor()
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, vit_processor, extract_lbp_features
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate model
    evaluate_model(model, test_loader, vit_processor, extract_lbp_features, class_names)

if __name__ == "__main__":
    main()
