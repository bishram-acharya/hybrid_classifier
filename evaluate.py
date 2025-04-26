"""
Evaluation functionality for the hybrid classifier.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from .models.hybridnet import prepare_batch
from .utils.visualization import plot_confusion_matrix

def evaluate_model(model, test_loader, vit_processor, extract_lbp_fn, class_names):
    """
    Evaluate the trained model on test data (binary classification).
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img_tensor, vit_inputs, lbp_feat, labels = prepare_batch(
                batch, vit_processor, extract_lbp_fn
            )
            
            outputs = model(img_tensor, vit_inputs, lbp_feat)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            _, preds = torch.max(probs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Calculate AUC and plot ROC curve
    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])  # Probability of positive class
        print(f"\nAUC: {roc_auc:.4f}")
        
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"ROC AUC calculation failed: {e}")
    
    return y_true, y_pred
