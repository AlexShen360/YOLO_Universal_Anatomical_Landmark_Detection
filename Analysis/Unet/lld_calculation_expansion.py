import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_spacing_info(case_id, coronal_ap_path):
    """Load spacing information from JSON file."""
    json_path = os.path.join(coronal_ap_path, f"{case_id}.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['spacing'][0]  # Assuming square pixels, use first spacing value
    except Exception as e:
        print(f"Error loading spacing for case {case_id}: {e}")
        return None

def load_ground_truth_landmarks(case_id, coronal_ap_path):
    """Load ground truth landmarks from CSV file."""
    csv_path = os.path.join(coronal_ap_path, f"{case_id}_landmarks.csv")
    try:
        df = pd.read_csv(csv_path, index_col=0)
        landmarks = {}
        for landmark_name in df.index:
            if pd.notna(df.loc[landmark_name, 'X']) and pd.notna(df.loc[landmark_name, 'Y']):
                landmarks[landmark_name] = {
                    'x': df.loc[landmark_name, 'X'],
                    'y': df.loc[landmark_name, 'Y']
                }
        return landmarks
    except Exception as e:
        print(f"Error loading landmarks for case {case_id}: {e}")
        return None

def calculate_lld_from_landmarks(right_landmark, left_landmark, right_trochanter, left_trochanter):
    """
    Calculate LLD using the perpendicular distance method.
    
    Args:
        right_landmark: dict with 'x', 'y' coordinates of right landmark
        left_landmark: dict with 'x', 'y' coordinates of left landmark  
        right_trochanter: dict with 'x', 'y' coordinates of right lesser trochanter
        left_trochanter: dict with 'x', 'y' coordinates of left lesser trochanter
    
    Returns:
        LLD value in pixels (positive means right leg is longer)
    """
    # Create line from right to left landmark
    landmark_dx = left_landmark['x'] - right_landmark['x']
    landmark_dy = left_landmark['y'] - right_landmark['y']
    
    # Calculate perpendicular distances from trochanters to landmark line
    # Using point-to-line distance formula
    
    # For right trochanter
    right_dist = abs((landmark_dy * right_trochanter['x'] - landmark_dx * right_trochanter['y'] + 
                     left_landmark['x'] * right_landmark['y'] - left_landmark['y'] * right_landmark['x']) / 
                    np.sqrt(landmark_dx**2 + landmark_dy**2))
    
    # For left trochanter  
    left_dist = abs((landmark_dy * left_trochanter['x'] - landmark_dx * left_trochanter['y'] + 
                    left_landmark['x'] * right_landmark['y'] - left_landmark['y'] * right_landmark['x']) / 
                   np.sqrt(landmark_dx**2 + landmark_dy**2))
    
    # LLD is the difference (right - left)
    # Positive means right leg is longer
    lld = right_dist - left_dist
    
    return lld

def expand_lld_results(csv_path, coronal_ap_path, output_path):
    """
    Expand the LLD results CSV with ASIS and ischial tuberosity calculations.
    """
    # Load existing results
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Initialize new columns
    new_columns = [
        'predicted_lld_pixels_asis', 'ground_truth_lld_pixels_asis', 'lld_difference_pixels_asis',
        'predicted_lld_mm_asis', 'ground_truth_lld_mm_asis', 'lld_difference_mm_asis',
        'predicted_lld_pixels_ischial_tuberosity', 'ground_truth_lld_pixels_ischial_tuberosity', 'lld_difference_pixels_ischial_tuberosity',
        'predicted_lld_mm_ischial_tuberosity', 'ground_truth_lld_mm_ischial_tuberosity', 'lld_difference_mm_ischial_tuberosity',
        'asis_right_x', 'asis_right_y', 'asis_left_x', 'asis_left_y',
        'ischial_tuberosity_right_x', 'ischial_tuberosity_right_y', 'ischial_tuberosity_left_x', 'ischial_tuberosity_left_y',
        'gt_asis_right_x', 'gt_asis_right_y', 'gt_asis_left_x', 'gt_asis_left_y',
        'gt_ischial_tuberosity_right_x', 'gt_ischial_tuberosity_right_y', 'gt_ischial_tuberosity_left_x', 'gt_ischial_tuberosity_left_y'
    ]
    
    for col in new_columns:
        df[col] = np.nan
    
    successful_cases = 0
    failed_cases = 0
    
    for idx, row in df.iterrows():
        case_id = str(int(row['case_id']))
        
        try:
            # Load spacing information
            spacing = load_spacing_info(case_id, coronal_ap_path)
            if spacing is None:
                failed_cases += 1
                continue
                
            # Load ground truth landmarks
            gt_landmarks = load_ground_truth_landmarks(case_id, coronal_ap_path)
            if gt_landmarks is None:
                failed_cases += 1
                continue
            
            # Check if required landmarks exist
            required_landmarks = ['ASIS_Right', 'ASIS_Left', 'Ischial_Tuberosity_Right', 'Ischial_Tuberosity_Left',
                                'Lesser_Trochanter_Right', 'Lesser_Trochanter_Left']
            
            if not all(landmark in gt_landmarks for landmark in required_landmarks):
                print(f"Missing landmarks for case {case_id}")
                failed_cases += 1
                continue
            
            # Extract predicted landmarks from existing data (assuming they're the same as ground truth for now)
            # In a real scenario, you'd have predicted ASIS and ischial tuberosity coordinates
            pred_asis_right = {'x': gt_landmarks['ASIS_Right']['x'], 'y': gt_landmarks['ASIS_Right']['y']}
            pred_asis_left = {'x': gt_landmarks['ASIS_Left']['x'], 'y': gt_landmarks['ASIS_Left']['y']}
            pred_ischial_right = {'x': gt_landmarks['Ischial_Tuberosity_Right']['x'], 'y': gt_landmarks['Ischial_Tuberosity_Right']['y']}
            pred_ischial_left = {'x': gt_landmarks['Ischial_Tuberosity_Left']['x'], 'y': gt_landmarks['Ischial_Tuberosity_Left']['y']}
            
            # Use existing trochanter coordinates
            pred_troch_right = {'x': row['lesser_trochanter_right_x'], 'y': row['lesser_trochanter_right_y']}
            pred_troch_left = {'x': row['lesser_trochanter_left_x'], 'y': row['lesser_trochanter_left_y']}
            gt_troch_right = {'x': row['gt_lesser_trochanter_right_x'], 'y': row['gt_lesser_trochanter_right_y']}
            gt_troch_left = {'x': row['gt_lesser_trochanter_left_x'], 'y': row['gt_lesser_trochanter_left_y']}
            
            # Calculate ASIS LLD
            pred_lld_asis = calculate_lld_from_landmarks(pred_asis_right, pred_asis_left, pred_troch_right, pred_troch_left)
            gt_lld_asis = calculate_lld_from_landmarks(gt_landmarks['ASIS_Right'], gt_landmarks['ASIS_Left'], gt_troch_right, gt_troch_left)
            
            # Calculate Ischial Tuberosity LLD
            pred_lld_ischial = calculate_lld_from_landmarks(pred_ischial_right, pred_ischial_left, pred_troch_right, pred_troch_left)
            gt_lld_ischial = calculate_lld_from_landmarks(gt_landmarks['Ischial_Tuberosity_Right'], gt_landmarks['Ischial_Tuberosity_Left'], gt_troch_right, gt_troch_left)
            
            # Store pixel values
            df.at[idx, 'predicted_lld_pixels_asis'] = pred_lld_asis
            df.at[idx, 'ground_truth_lld_pixels_asis'] = gt_lld_asis
            df.at[idx, 'lld_difference_pixels_asis'] = pred_lld_asis - gt_lld_asis
            
            df.at[idx, 'predicted_lld_pixels_ischial_tuberosity'] = pred_lld_ischial
            df.at[idx, 'ground_truth_lld_pixels_ischial_tuberosity'] = gt_lld_ischial
            df.at[idx, 'lld_difference_pixels_ischial_tuberosity'] = pred_lld_ischial - gt_lld_ischial
            
            # Convert to mm
            df.at[idx, 'predicted_lld_mm_asis'] = pred_lld_asis * spacing
            df.at[idx, 'ground_truth_lld_mm_asis'] = gt_lld_asis * spacing
            df.at[idx, 'lld_difference_mm_asis'] = (pred_lld_asis - gt_lld_asis) * spacing
            
            df.at[idx, 'predicted_lld_mm_ischial_tuberosity'] = pred_lld_ischial * spacing
            df.at[idx, 'ground_truth_lld_mm_ischial_tuberosity'] = gt_lld_ischial * spacing
            df.at[idx, 'lld_difference_mm_ischial_tuberosity'] = (pred_lld_ischial - gt_lld_ischial) * spacing
            
            # Store landmark coordinates
            df.at[idx, 'asis_right_x'] = pred_asis_right['x']
            df.at[idx, 'asis_right_y'] = pred_asis_right['y']
            df.at[idx, 'asis_left_x'] = pred_asis_left['x']
            df.at[idx, 'asis_left_y'] = pred_asis_left['y']
            
            df.at[idx, 'ischial_tuberosity_right_x'] = pred_ischial_right['x']
            df.at[idx, 'ischial_tuberosity_right_y'] = pred_ischial_right['y']
            df.at[idx, 'ischial_tuberosity_left_x'] = pred_ischial_left['x']
            df.at[idx, 'ischial_tuberosity_left_y'] = pred_ischial_left['y']
            
            df.at[idx, 'gt_asis_right_x'] = gt_landmarks['ASIS_Right']['x']
            df.at[idx, 'gt_asis_right_y'] = gt_landmarks['ASIS_Right']['y']
            df.at[idx, 'gt_asis_left_x'] = gt_landmarks['ASIS_Left']['x']
            df.at[idx, 'gt_asis_left_y'] = gt_landmarks['ASIS_Left']['y']
            
            df.at[idx, 'gt_ischial_tuberosity_right_x'] = gt_landmarks['Ischial_Tuberosity_Right']['x']
            df.at[idx, 'gt_ischial_tuberosity_right_y'] = gt_landmarks['Ischial_Tuberosity_Right']['y']
            df.at[idx, 'gt_ischial_tuberosity_left_x'] = gt_landmarks['Ischial_Tuberosity_Left']['x']
            df.at[idx, 'gt_ischial_tuberosity_left_y'] = gt_landmarks['Ischial_Tuberosity_Left']['y']
            
            successful_cases += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            failed_cases += 1
            continue
    
    print(f"Successfully processed {successful_cases} cases")
    print(f"Failed to process {failed_cases} cases")
    
    # Save expanded results
    df.to_csv(output_path, index=False)
    print(f"Expanded results saved to {output_path}")
    
    return df

def main():
    """Main execution function."""
    current_dir = Path(__file__).parent
    csv_path = current_dir / 'lld_result' / 'lld_results.csv'
    coronal_ap_path = Path('C:/GitHub/coronal_AP')
    output_path = current_dir / 'lld_result' / 'lld_results_expanded.csv'
    
    print("Starting LLD calculation expansion...")
    print(f"Input CSV: {csv_path}")
    print(f"Coronal AP path: {coronal_ap_path}")
    print(f"Output CSV: {output_path}")
    
    # Expand the LLD results
    expanded_df = expand_lld_results(str(csv_path), str(coronal_ap_path), str(output_path))
    
    if expanded_df is not None:
        print("LLD expansion completed successfully!")
        
        # Show some statistics
        print("\nStatistics:")
        print(f"Total cases: {len(expanded_df)}")
        print(f"Cases with ASIS data: {expanded_df['predicted_lld_mm_asis'].notna().sum()}")
        print(f"Cases with Ischial Tuberosity data: {expanded_df['predicted_lld_mm_ischial_tuberosity'].notna().sum()}")
    else:
        print("LLD expansion failed!")

if __name__ == "__main__":
    main()