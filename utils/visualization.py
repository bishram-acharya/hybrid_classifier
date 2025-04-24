"""
Visualization utilities for model analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot the training history.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accuracies (list): Training accuracies per epoch
        val_accuracies (list): Validation accuracies per epoch
    """
    num_epochs = len(train_losses)
    plt.figure(figsize=(12, 5))
    
    # Loss curve plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    
    # Accuracy curve plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Acc')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): Names of classes
    """
    conf = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()