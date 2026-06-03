import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path

# Categorize LLD values into three categories (matches roc_analysis_expanded.py)
def categorize_lld(lld_values):
    abs_lld = np.abs(lld_values)
    categories = np.zeros(len(abs_lld))
    categories[(abs_lld >= 5) & (abs_lld < 10)] = 1
    categories[abs_lld >= 10] = 2
    return categories.astype(int)

def create_roc_plot(y_true, y_pred_continuous, method_label, output_path):
    """
    Creates an ROC plot for a specific LLD measurement method.
    Using absolute values as requested.
    """
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred_continuous)
    
    y_true_cat = categorize_lld(abs_true)
    
    n_classes = 3
    class_names = ['0-5mm', '5-10mm', '>10mm']
    y_true_bin = label_binarize(y_true_cat, classes=[0, 1, 2])
    
    # Create probability-like scores (following roc_analysis_expanded.py)
    y_score = np.zeros((len(abs_pred), n_classes))
    
    # Class 0 (0-5mm)
    y_score[:, 0] = np.maximum(0, 1 - np.maximum(0, abs_pred - 5) / 10)
    # Class 1 (5-10mm)
    dist_to_class1 = np.minimum(np.abs(abs_pred - 5), np.abs(abs_pred - 10))
    y_score[:, 1] = np.maximum(0, 1 - dist_to_class1 / 5)
    # Class 2 (>10mm)
    y_score[:, 2] = np.maximum(0, (abs_pred - 10) / 20)
    y_score[:, 2] = np.minimum(1, y_score[:, 2])
    
    # Normalize
    row_sums = y_score.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_score = y_score / row_sums[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green']
    
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:.3f})')
        else:
            print(f"Warning: No samples for class {class_names[i]} in {method_label}")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average ROC (AUC = {auc_micro:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Hip LLD ({method_label})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC plot saved to {output_path}")

def main():
    # Use absolute paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "Analysis" / "hip_evaluation_results.csv"
    output_dir = project_root / "Analysis" / "roc_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    methods = {
        'ASIS': ('gt_lld_asis_mm', 'pred_lld_asis_mm'),
        'Teardrop': ('gt_lld_teardrop_mm', 'pred_lld_teardrop_mm'),
        'Ischial Tuberosity': ('gt_lld_ischial_mm', 'pred_lld_ischial_mm')
    }
    
    for label, (gt_col, pred_col) in methods.items():
        if gt_col in df.columns and pred_col in df.columns:
            y_true = df[gt_col].values
            y_pred = df[pred_col].values
            
            output_filename = f"hip_lld_roc_{label.lower().replace(' ', '_')}.png"
            output_path = output_dir / output_filename
            
            create_roc_plot(y_true, y_pred, label, output_path)
        else:
            print(f"Warning: Columns {gt_col} or {pred_col} not found in CSV.")

if __name__ == "__main__":
    main()
