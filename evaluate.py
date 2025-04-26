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
    Prints 5 correct and 5 wrong prediction examples from each class.
    """
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
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            _, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

            # Store examples
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

    # Helper function to plot examples
    def plot_examples_grid(correct_examples, wrong_examples, class_names):
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
                    ax.imshow(img)
                    ax.set_title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]} ({prob:.2f})", fontsize=10)
                ax.axis('off')
            
            # Add row title to the left
            axes[row, 0].set_ylabel(row_titles[row], fontsize=14, rotation=0, labelpad=70, va='center')

        plt.tight_layout()
        plt.show()


    # Plot examples
    print("\nCorrect Predictions Examples:")
    plot_examples_grid(correct_examples, "Correct")

    print("\nWrong Predictions Examples:")
    plot_examples_grid(wrong_examples, "Wrong")

    return y_true, y_pred
