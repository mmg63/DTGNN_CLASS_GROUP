# evaluate_metrics.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    roc_curve,
    auc
)

def evaluate_multilabel_metrics(y_true, y_pred, label_names=None):
    """
    Computes and prints classification metrics (precision, recall, F1).
    Returns multilabel confusion matrices.
    """
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    print("\n📊 Classification Report (Macro Averages):")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return multilabel_confusion_matrix(y_true, y_pred)

def plot_confusion_matrices(conf_matrices, class_names):
    """
    Plot confusion matrices for each class.
    """
    for i, cm in enumerate(conf_matrices):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for class: {class_names[i]}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

def plot_roc_curves(y_true, y_probs, class_names):
    """
    Plot ROC curves and AUC scores for each class.
    """
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_probs = y_probs.cpu().numpy() if isinstance(y_probs, torch.Tensor) else y_probs

    plt.figure(figsize=(8, 6))
    for i in range(y_true.shape[1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_multi_label_metrics(y_true, y_pred, threshold=0.5):
    """
    y_true and y_pred should be tensors of shape [batch_size, num_classes]
    """
    y_prob = torch.sigmoid(y_pred)  # Apply sigmoid
    y_pred_bin = (y_prob > threshold).int()
    y_true = y_true.int()

    # Convert to numpy for sklearn
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred_bin.cpu().numpy()

    metrics = {
        'exact_match_acc': (y_pred_bin == y_true).all(dim=1).float().mean().item(),
        'soft_accuracy': (y_pred_bin == y_true).float().mean().item(),
        'macro_precision': precision_score(y_true_np, y_pred_np, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true_np, y_pred_np, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0),
    }

    return metrics
