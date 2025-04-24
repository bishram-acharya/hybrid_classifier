"""
Training procedures for the hybrid classifier.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .models.hybridnet import prepare_batch
from .config import LEARNING_RATE, WEIGHT_DECAY, DEVICE, NUM_EPOCHS
from .utils.visualization import plot_training_history

def train_model(model, train_loader, val_loader, vit_processor, extract_lbp_fn, num_epochs=NUM_EPOCHS):
    """
    Train the hybrid model.
    
    Args:
        model (HybridNet): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        vit_processor (ViTImageProcessor): Processor for Vision Transformer
        extract_lbp_fn (callable): Function to extract LBP features
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (train_losses, val_losses, train_accuracies, val_accuracies)
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Training with validation tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            img_tensor, vit_inputs, lbp_feat, labels = prepare_batch(
                batch, vit_processor, extract_lbp_fn
            )
            
            optimizer.zero_grad()
            outputs = model(img_tensor, vit_inputs, lbp_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img_tensor.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                img_tensor, vit_inputs, lbp_feat, labels = prepare_batch(
                    batch, vit_processor, extract_lbp_fn
                )
                
                outputs = model(img_tensor, vit_inputs, lbp_feat)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * img_tensor.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)
        
        print(f"Epoch {epoch+1}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Acc: {val_accuracies[-1]:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies
