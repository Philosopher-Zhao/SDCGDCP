def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x
import pandas as pd
import os
from config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def save_results(epoch, train_metrics, val_metrics, best_val_loss=float('inf')):
    res = {
        'epoch': epoch,
        'train_loss': train_metrics['loss'],
        'train_auc': train_metrics['auc'],
        'train_ap': train_metrics.get('ap', 0),
        'val_loss': val_metrics['loss'],
        'val_auc': val_metrics['auc'],
        'val_ap': val_metrics.get('ap', 0),
        'val_threshold': val_metrics.get('threshold', 0.5),
        'val_precision': val_metrics.get('precision', 0),
        'val_recall': val_metrics.get('recall', 0),
        'val_f1': val_metrics.get('f1', 0),
        'is_best': val_metrics.get('loss', float('inf')) < best_val_loss
    }

    df = pd.DataFrame([res])
    header = not os.path.exists(Config.result_file)
    df.to_csv(Config.result_file, mode='a', header=header, index=False)

def plot_roc_curve(labels, probs, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(labels, probs, save_path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="upper right")
    plt.savefig(save_path)
    plt.close()