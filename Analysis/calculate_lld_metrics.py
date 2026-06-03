import pandas as pd
import numpy as np
from pathlib import Path

def calculate_metrics(df, method_name, gt_col, pred_col):
    """
    Calculate Mean Absolute Error (MAE) and Mean Deviation (MD/Bias).
    """
    if gt_col not in df.columns or pred_col not in df.columns:
        return None
    
    # Calculate difference
    diff = df[pred_col] - df[gt_col]
    
    mae = np.mean(np.abs(diff))
    md = np.mean(diff)
    std_diff = np.std(diff)
    
    return {
        'Method': method_name,
        'MAE (mm)': mae,
        'MD (mm)': md,
        'Std Dev (mm)': std_diff
    }

def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "Analysis" / "hip_evaluation_results.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}\n")
    
    methods = [
        ('ASIS', 'gt_lld_asis_mm', 'pred_lld_asis_mm'),
        ('Teardrop', 'gt_lld_teardrop_mm', 'pred_lld_teardrop_mm'),
        ('Ischial Tuberosity', 'gt_lld_ischial_mm', 'pred_lld_ischial_mm')
    ]
    
    results = []
    for name, gt, pred in methods:
        metrics = calculate_metrics(df, name, gt, pred)
        if metrics:
            results.append(metrics)
        else:
            print(f"Warning: Columns for {name} not found.")
            
    if results:
        results_df = pd.DataFrame(results)
        print("LLD Measurement Metrics (mm):")
        print("==============================")
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # Save results to a text file as well
        output_txt = project_root / "Analysis" / "lld_metrics_results.txt"
        with open(output_txt, 'w') as f:
            f.write("LLD Measurement Metrics (mm):\n")
            f.write("==============================\n")
            f.write(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nResults also saved to {output_txt}")

if __name__ == "__main__":
    main()
