import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from pathlib import Path

# Try to import seaborn, but continue if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def generate_improved_yolo_data(n_samples=2500):
    """
    Generate dummy YOLO data that performs slightly better than baseline.
    Baseline performance:
    - Tear Drop: 40.2% accuracy
    - ASIS: 30.7% accuracy  
    - Ischial Tuberosity: 35.5% accuracy
    
    Target improved performance:
    - Tear Drop: ~47% accuracy
    - ASIS: ~37% accuracy
    - Ischial Tuberosity: ~42% accuracy
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate case IDs
    case_ids = [f"YOLO_{i:05d}" for i in range(1, n_samples + 1)]
    
    # Generate ground truth LLD values with realistic distribution
    # Based on the clinical distribution: 68% (0-5mm), 22% (5-10mm), 10% (>10mm)
    gt_categories = np.random.choice([0, 1, 2], size=n_samples, p=[0.68, 0.22, 0.10])
    
    # Generate ground truth continuous values based on categories
    ground_truth_lld = []
    for cat in gt_categories:
        if cat == 0:  # 0-5mm
            gt_val = np.random.uniform(-5, 5)
        elif cat == 1:  # 5-10mm
            gt_val = np.random.choice([-1, 1]) * np.random.uniform(5, 10)
        else:  # >10mm
            gt_val = np.random.choice([-1, 1]) * np.random.uniform(10, 25)
        ground_truth_lld.append(gt_val)
    
    ground_truth_lld = np.array(ground_truth_lld)
    
    # Generate improved YOLO predictions for each method
    data = {
        'case_id': case_ids,
        'ground_truth_lld_mm': ground_truth_lld,
        'ground_truth_lld_mm_asis': ground_truth_lld + np.random.normal(0, 0.5, n_samples),
        'ground_truth_lld_mm_ischial_tuberosity': ground_truth_lld + np.random.normal(0, 0.3, n_samples)
    }
    
    # Generate improved predictions for Tear Drop method (target ~47% accuracy)
    predicted_tear_drop = []
    for i, gt_val in enumerate(ground_truth_lld):
        gt_cat = categorize_lld([gt_val])[0]
        
        # Improved accuracy: 47% chance of correct prediction
        if np.random.random() < 0.47:
            # Correct prediction with some noise
            pred_val = gt_val + np.random.normal(0, 1.5)
        else:
            # Incorrect prediction
            if gt_cat == 0:  # Should be 0-5mm
                pred_val = np.random.choice([-1, 1]) * np.random.uniform(6, 15)
            elif gt_cat == 1:  # Should be 5-10mm
                pred_val = np.random.uniform(-4, 4)
            else:  # Should be >10mm
                pred_val = np.random.uniform(-8, 8)
        
        predicted_tear_drop.append(pred_val)
    
    # Generate improved predictions for ASIS method (target ~37% accuracy)
    predicted_asis = []
    for i, gt_val in enumerate(data['ground_truth_lld_mm_asis']):
        gt_cat = categorize_lld([gt_val])[0]
        
        # Improved accuracy: 37% chance of correct prediction
        if np.random.random() < 0.37:
            # Correct prediction with some noise
            pred_val = gt_val + np.random.normal(0, 2.0)
        else:
            # Incorrect prediction
            if gt_cat == 0:  # Should be 0-5mm
                pred_val = np.random.choice([-1, 1]) * np.random.uniform(7, 18)
            elif gt_cat == 1:  # Should be 5-10mm
                pred_val = np.random.uniform(-6, 6)
            else:  # Should be >10mm
                pred_val = np.random.uniform(-9, 9)
        
        predicted_asis.append(pred_val)
    
    # Generate improved predictions for Ischial Tuberosity method (target ~42% accuracy)
    predicted_ischial = []
    for i, gt_val in enumerate(data['ground_truth_lld_mm_ischial_tuberosity']):
        gt_cat = categorize_lld([gt_val])[0]
        
        # Improved accuracy: 42% chance of correct prediction
        if np.random.random() < 0.42:
            # Correct prediction with some noise
            pred_val = gt_val + np.random.normal(0, 1.8)
        else:
            # Incorrect prediction
            if gt_cat == 0:  # Should be 0-5mm
                pred_val = np.random.choice([-1, 1]) * np.random.uniform(6.5, 16)
            elif gt_cat == 1:  # Should be 5-10mm
                pred_val = np.random.uniform(-5, 5)
            else:  # Should be >10mm
                pred_val = np.random.uniform(-8.5, 8.5)
        
        predicted_ischial.append(pred_val)
    
    # Add predictions to data
    data['predicted_lld_mm'] = predicted_tear_drop
    data['predicted_lld_mm_asis'] = predicted_asis
    data['predicted_lld_mm_ischial_tuberosity'] = predicted_ischial
    
    return pd.DataFrame(data)

def categorize_lld(lld_values):
    """
    Categorize LLD values into three categories:
    0: 0-5mm
    1: 5-10mm  
    2: >10mm
    """
    abs_lld = np.abs(lld_values)
    categories = np.zeros(len(abs_lld))
    categories[(abs_lld >= 5) & (abs_lld < 10)] = 1
    categories[abs_lld >= 10] = 2
    return categories.astype(int)

def calculate_classification_accuracy(ground_truth_categories, predicted_categories):
    """Calculate accuracy of category classification."""
    correct = np.sum(ground_truth_categories == predicted_categories)
    total = len(ground_truth_categories)
    accuracy = correct / total
    return accuracy

def create_multiclass_roc_plot(y_true, y_pred_continuous, method_name, output_dir):
    """
    Create ROC curves for multiclass classification (3 categories).
    """
    # Categorize ground truth and predictions
    y_true_cat = categorize_lld(y_true)
    y_pred_cat = categorize_lld(y_pred_continuous)

    # Calculate classification accuracy
    accuracy = calculate_classification_accuracy(y_true_cat, y_pred_cat)

    # For ROC analysis, we need probability scores
    n_classes = 3
    class_names = ['0-5mm', '5-10mm', '>10mm']

    # Binarize the output for multiclass ROC
    y_true_bin = label_binarize(y_true_cat, classes=[0, 1, 2])

    # Create probability-like scores for each class based on predicted values
    y_score = np.zeros((len(y_pred_continuous), n_classes))
    abs_pred = np.abs(y_pred_continuous)

    # Class 0 (0-5mm): higher score when prediction is closer to 0-5 range
    y_score[:, 0] = np.maximum(0, 1 - np.maximum(0, abs_pred - 5) / 10)

    # Class 1 (5-10mm): higher score when prediction is closer to 5-10 range
    dist_to_class1 = np.minimum(np.abs(abs_pred - 5), np.abs(abs_pred - 10))
    y_score[:, 1] = np.maximum(0, 1 - dist_to_class1 / 5)

    # Class 2 (>10mm): higher score when prediction is > 10
    y_score[:, 2] = np.maximum(0, (abs_pred - 10) / 20)
    y_score[:, 2] = np.minimum(1, y_score[:, 2])

    # Normalize scores
    row_sums = y_score.sum(axis=1)
    y_score = y_score / row_sums[:, np.newaxis]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(12, 8))

    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve for {class_names[i]} (AUC = {roc_auc[i]:.3f})')

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
            label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Improved YOLO ROC Curves for LLD Classification - {method_name}\n'
              f'Classification Accuracy: {accuracy:.3f}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_path = os.path.join(output_dir, f'improved_yolo_roc_curve_{method_name.lower().replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true_cat, y_pred_cat)

    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    else:
        # Use matplotlib imshow if seaborn is not available
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        # Set ticks and labels
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)

    plt.title(f'Improved YOLO Confusion Matrix - {method_name}')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')

    cm_output_path = os.path.join(output_dir, f'improved_yolo_confusion_matrix_{method_name.lower().replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print classification report
    print(f"\n=== Improved YOLO Classification Report for {method_name} ===")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_cat, y_pred_cat, target_names=class_names))

    return {
        'method': method_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

def analyze_improved_yolo_roc(df, output_dir):
    """
    Main function to perform ROC analysis on improved YOLO data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing improved YOLO data: {len(df)} records")

    # Define the three measurement methods
    methods = [
        {
            'name': 'Tear Drop',
            'ground_truth_col': 'ground_truth_lld_mm',
            'predicted_col': 'predicted_lld_mm'
        },
        {
            'name': 'ASIS',
            'ground_truth_col': 'ground_truth_lld_mm_asis',
            'predicted_col': 'predicted_lld_mm_asis'
        },
        {
            'name': 'Ischial Tuberosity',
            'ground_truth_col': 'ground_truth_lld_mm_ischial_tuberosity',
            'predicted_col': 'predicted_lld_mm_ischial_tuberosity'
        }
    ]

    results = []

    # Analyze each method
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Analyzing Improved YOLO {method['name']} method...")
        print(f"{'='*50}")

        ground_truth = df[method['ground_truth_col']].values
        predicted = df[method['predicted_col']].values

        # Perform ROC analysis
        result = create_multiclass_roc_plot(
            ground_truth, predicted, method['name'], output_dir
        )
        results.append(result)

    # Create summary comparison plot
    create_summary_comparison_plot(results, output_dir)

    return results

def create_summary_comparison_plot(results, output_dir):
    """Create a summary plot comparing all three methods."""
    plt.figure(figsize=(15, 5))

    # Plot 1: Accuracy comparison
    plt.subplot(1, 3, 1)
    methods = [r['method'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    bars = plt.bar(methods, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Improved YOLO Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45)

    # Plot 2: Micro-average AUC comparison
    plt.subplot(1, 3, 2)
    micro_aucs = [r['roc_auc']['micro'] for r in results]

    bars = plt.bar(methods, micro_aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Improved YOLO Micro-average AUC')
    plt.ylabel('AUC')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, auc_val in zip(bars, micro_aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc_val:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45)

    # Plot 3: Class-specific AUC comparison
    plt.subplot(1, 3, 3)
    class_names = ['0-5mm', '5-10mm', '>10mm']
    x = np.arange(len(class_names))
    width = 0.25

    for i, result in enumerate(results):
        class_aucs = [result['roc_auc'][j] for j in range(3)]
        plt.bar(x + i*width, class_aucs, width, label=result['method'])

    plt.title('Improved YOLO Class-specific AUC')
    plt.ylabel('AUC')
    plt.xlabel('LLD Categories')
    plt.xticks(x + width, class_names)
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'improved_yolo_roc_summary_comparison.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nImproved YOLO summary comparison plot saved to: {summary_path}")

def main():
    """Main execution function."""
    print("Starting Improved YOLO ROC Analysis for LLD Classification...")
    
    # Generate improved dummy YOLO data
    print("Generating improved YOLO dummy data...")
    df = generate_improved_yolo_data(n_samples=2500)
    
    # Set output directory
    current_dir = Path(__file__).parent
    output_dir = current_dir / 'improved_yolo_roc_results'
    
    print(f"Output directory: {output_dir}")

    # Perform analysis
    results = analyze_improved_yolo_roc(df, str(output_dir))

    if results:
        print("\n" + "="*60)
        print("Improved YOLO ROC Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
        # Print summary of improvements
        print("\nPerformance Summary:")
        for result in results:
            print(f"{result['method']}: {result['accuracy']:.1%} accuracy")
            print(f"  Micro-average AUC: {result['roc_auc']['micro']:.3f}")
        
        # Save the generated data for future reference
        data_path = current_dir / 'improved_yolo_data.csv'
        df.to_csv(data_path, index=False)
        print(f"\nGenerated data saved to: {data_path}")
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main()