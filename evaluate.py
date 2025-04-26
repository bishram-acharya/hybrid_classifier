import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from .models.hybridnet import prepare_batch
from .utils.visualization import plot_confusion_matrix
from .utils.gradcam import GradCAM

def plot_examples_grid(correct_examples, wrong_examples, class_names):
    """
    Plot a grid of examples showing correct and incorrect predictions.
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # 4 rows, 5 columns
    fig.subplots_adjust(hspace=0.4)
    
    row_titles = [
        f"Correct - {class_names[0]}",
        f"Wrong - {class_names[0]}",
        f"Correct - {class_names[1]}",
        f"Wrong - {class_names[1]}"
    ]
    
    for row in range(4):
        if row == 0:
            examples = correct_examples[0]
        elif row == 1:
            examples = wrong_examples[0]
        elif row == 2:
            examples = correct_examples[1]
        elif row == 3:
            examples = wrong_examples[1]
        
        for col in range(5):
            ax = axes[row, col]
            if col < len(examples):
                img, true_label, pred_label, prob = examples[col]
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)  # Prevent strange colors
                ax.imshow(img)
                ax.set_title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]} ({prob:.2f})", fontsize=10)
            ax.axis('off')
        
        axes[row, 0].set_ylabel(row_titles[row], fontsize=14, rotation=0, labelpad=70, va='center')

    plt.tight_layout()
    plt.show()

def plot_examples_with_gradcam(correct_examples, wrong_examples, class_names, model, vit_processor, extract_lbp_fn, device):
    """
    Plot examples with GradCAM visualizations in a grid
    """
    # Initialize GradCAM
    grad_cam = GradCAM(model)
    
    # Create figure with 4 rows (2 classes x correct/wrong) and 5 columns
    # Each column has 3 images (original, heatmap, overlay)
    fig, axes = plt.subplots(4, 15, figsize=(25, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    
    row_titles = [
        f"Correct - {class_names[0]}",
        f"Wrong - {class_names[0]}",
        f"Correct - {class_names[1]}",
        f"Wrong - {class_names[1]}"
    ]
    
    # Process each row
    for row in range(4):
        if row == 0:
            examples = correct_examples[0]
            title = f"Correct - {class_names[0]}"
        elif row == 1:
            examples = wrong_examples[0]
            title = f"Wrong - {class_names[0]}"
        elif row == 2:
            examples = correct_examples[1]
            title = f"Correct - {class_names[1]}"
        elif row == 3:
            examples = wrong_examples[1]
            title = f"Wrong - {class_names[1]}"
        
        # For each example in the row (max 5)
        for col in range(5):
            if col < len(examples):
                img_tensor, true_label, pred_label, prob = examples[col]
                
                # Generate GradCAM visualizations
                orig, heatmap, overlay = grad_cam.process_example(
                    img_tensor, 
                    pred_label, 
                    model, 
                    vit_processor, 
                    extract_lbp_fn, 
                    device
                )
                
                # Plot original image
                ax = axes[row, col*3]
                ax.imshow(orig)
                if row == 0:
                    ax.set_title(f"Example {col+1}", fontsize=10)
                ax.set_xlabel(f"T:{class_names[true_label]}\nP:{class_names[pred_label]} ({prob:.2f})", fontsize=8)
                ax.axis('off')
                
                # Plot heatmap
                ax = axes[row, col*3+1]
                ax.imshow(heatmap)
                if row == 0:
                    ax.set_title("Heatmap", fontsize=10)
                ax.axis('off')
                
                # Plot overlay
                ax = axes[row, col*3+2]
                ax.imshow(overlay)
                if row == 0:
                    ax.set_title("Overlay", fontsize=10)
                ax.axis('off')
            else:
                # If we have fewer than 5 examples, turn off the extra axes
                for i in range(3):
                    axes[row, col*3+i].axis('off')
        
        # Set row labels
        axes[row, 0].set_ylabel(row_titles[row], fontsize=12, rotation=0, labelpad=70, va='center')

    plt.tight_layout()
    plt.show()
    
    # Clean up hooks to prevent memory leaks
    grad_cam.remove_hooks()

def evaluate_model(model, test_loader, vit_processor, extract_lbp_fn, class_names):
    """
    Evaluate the trained model on test data (binary classification).
    Prints 5 correct and 5 wrong prediction examples from each class.
    Generates GradCAM visualizations for predictions.
    """
    # Determine device
    device = next(model.parameters()).device
    
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    # Initialize dictionaries
    correct_examples = {0: [], 1: []}
    wrong_examples = {0: [], 1: []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img_tensor, vit_inputs, lbp_feat, labels = prepare_batch(
                batch, vit_processor, extract_lbp_fn
            )

            outputs = model(img_tensor, vit_inputs, lbp_feat)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = probs[i][pred_label].item()
                example = (img_tensor[i], true_label, pred_label, confidence)

                if true_label == pred_label:
                    if len(correct_examples[true_label]) < 5:
                        correct_examples[true_label].append(example)
                else:
                    if len(wrong_examples[true_label]) < 5:
                        wrong_examples[true_label].append(example)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names)

    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
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

    print("\nPlotting regular examples in a grid:")
    plot_examples_grid(correct_examples, wrong_examples, class_names)
    
    print("\nPlotting examples with GradCAM visualizations:")
    plot_examples_with_gradcam(
        correct_examples, 
        wrong_examples, 
        class_names, 
        model,
        vit_processor, 
        extract_lbp_fn, 
        device
    )

    return y_true, y_pred