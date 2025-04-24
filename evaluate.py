"""
Evaluation functionality for the hybrid classifier.
"""
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from .models.hybridnet import prepare_batch
from .utils.visualization import plot_confusion_matrix

def evaluate_model(model, test_loader, vit_processor, extract_lbp_fn, class_names):
    """
    Evaluate the trained model on test data.
    
    Args:
        model (HybridNet): The trained model
        test_loader (DataLoader): Test data loader
        vit_processor (ViTImageProcessor): Processor for Vision Transformer
        extract_lbp_fn (callable): Function to extract LBP features
        class_names (list): List of class names
        
    Returns:
        tuple: (y_true, y_pred) true and predicted labels
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img_tensor, vit_inputs, lbp_feat, labels = prepare_batch(
                batch, vit_processor, extract_lbp_fn
            )
            
            outputs = model(img_tensor, vit_inputs, lbp_feat)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    return y_true, y_pred