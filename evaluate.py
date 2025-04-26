"""
Evaluation functionality for the hybrid classifier.
"""
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from .models.hybridnet import prepare_batch
from .utils.visualization import plot_confusion_matrix

def plot_roc(y_true, y_score, class_names):
    """
    Plot ROC curves for each class.
    """
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def evaluate_model(model, test_loader, vit_processor, extract_lbp_fn, class_names):
    """
    Evaluate the trained model on test data.
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
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Calculate and plot AUC
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    y_probs = torch.tensor(y_probs).numpy()
    
    try:
        roc_auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
        print(f"\nMacro AUC: {roc_auc:.4f}")
        
        plot_roc(y_true_bin, y_probs, class_names)
    except Exception as e:
        print(f"ROC AUC calculation failed: {e}")
    
    return y_true, y_pred
