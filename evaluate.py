import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from .models.hybridnet import prepare_batch
from .utils.visualization import plot_confusion_matrix
from .utils.gradcam import MultiLayerGradCAM  # Updated import

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

def plot_examples_with_multi_layer_gradcam(correct_examples, wrong_examples, class_names, model, vit_processor, extract_lbp_fn, device):
    """
    Plot examples with GradCAM visualizations for multiple layers in a grid
    """
    # Initialize MultiLayerGradCAM
    multi_cam = MultiLayerGradCAM(model)
    
    # For each category (correct/wrong for each class)
    for category_idx, (category_name, examples_dict) in enumerate([
        ("Correct Examples", correct_examples),
        ("Wrong Examples", wrong_examples)
    ]):
        # For each class
        for class_idx, class_name in enumerate(class_names):
            examples = examples_dict[class_idx]
            if not examples:
                continue
                
            # For each example in this category and class
            for ex_idx, (img_tensor, true_label, pred_label, prob) in enumerate(examples):
                # Create a figure for this example with all layer visualizations
                fig = plt.figure(figsize=(20, 10))
                fig.suptitle(f"{category_name} - {class_name}: True={class_names[true_label]}, Pred={class_names[pred_label]} ({prob:.2f})", 
                             fontsize=16)
                
                # Process the example to get GradCAM visualizations for all layers
                orig_img, layer_results = multi_cam.process_example(
                    img_tensor, 
                    pred_label, 
                    model, 
                    vit_processor, 
                    extract_lbp_fn, 
                    device
                )
                
                # Total number of visualizations: original + (heatmap + overlay) for each layer
                num_layers = len(layer_results)
                
                # Create a 2-row grid: top row for original + heatmaps, bottom row for overlays
                gs = fig.add_gridspec(2, num_layers + 1)
                
                # Original image
                ax_orig = fig.add_subplot(gs[0, 0])
                ax_orig.imshow(orig_img)
                ax_orig.set_title('Original Image')
                ax_orig.axis('off')
                
                # Keep same position in bottom row empty for symmetry
                ax_empty = fig.add_subplot(gs[1, 0])
                ax_empty.axis('off')
                
                # Plot each layer's heatmap and overlay
                for i, (layer_name, (heatmap, overlay)) in enumerate(layer_results.items(), 1):
                    # Format layer name for display
                    display_name = layer_name.replace('_', ' ').title()
                    
                    # Heatmap (top row)
                    ax_heat = fig.add_subplot(gs[0, i])
                    ax_heat.imshow(heatmap)
                    ax_heat.set_title(f'{display_name}\nHeatmap')
                    ax_heat.axis('off')
                    
                    # Overlay (bottom row)
                    ax_over = fig.add_subplot(gs[1, i])
                    ax_over.imshow(overlay)
                    ax_over.set_title(f'{display_name}\nOverlay')
                    ax_over.axis('off')
                
                plt.tight_layout()
                plt.show()
    
    # Clean up hooks to prevent memory leaks
    multi_cam.remove_hooks()

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
    
    print("\nPlotting examples with multi-layer GradCAM visualizations:")
    plot_examples_with_multi_layer_gradcam(
        correct_examples, 
        wrong_examples, 
        class_names, 
        model,
        vit_processor, 
        extract_lbp_fn, 
        device
    )

    return y_true, y_pred