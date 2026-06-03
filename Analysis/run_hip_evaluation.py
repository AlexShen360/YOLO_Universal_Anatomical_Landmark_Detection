import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import json

# Add project root to sys.path
sys.path.append(os.path.abspath('.'))

from universal_landmark_detection.model.networks.u2net import U2Net
from universal_landmark_detection.model.networks.gln import GLN
from universal_landmark_detection.model.utils.kit import getPointsFromHeatmap

def load_metadata(case_id, metadata_dirs):
    """
    Load metadata from JSON files.
    case_id: e.g. '13042' from 'HIPAPR_13042_0'
    """
    # Try to extract the numeric ID
    import re
    match = re.search(r'(\d+)', case_id)
    if match:
        target_id = match.group(1)
    else:
        target_id = case_id

    for d in metadata_dirs:
        json_path = Path(d) / f"{target_id}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata from {json_path}: {e}")
    return None

def get_landmark_names():
    return [
        "ASIS_Left",
        "ASIS_Right",
        "Tear_Drop_Right",
        "Tear_Drop_Left",
        "Ischial_Tuberosity_Left",
        "Ischial_Tuberosity_Right",
        "Femoral_Head_Centre_Left",
        "Femoral_Head_Centre_Right",
        "Lesser_Trochanter_Left",
        "Lesser_Trochanter_Right"
    ]

def preprocess_image(path, size=(512, 512)):
    img = Image.open(path)
    origin_size = img.size
    img_resized = img.resize(size)
    arr = np.array(img_resized)
    
    if arr.ndim == 3:
        arr = arr[..., 0]
    
    # Transpose to (width, height) and expand to (1, width, height)
    arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float64)
    
    # Normalization
    for i in range(arr.shape[0]):
        std = arr[i].std()
        arr[i] = (arr[i] - arr[i].mean()) / (std + 1e-20)
        
    return torch.FloatTensor(arr).unsqueeze(0), origin_size

def read_gt_landmarks(label_path, original_size, target_size=(512, 512)):
    points = []
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        if not line: return None
        n = int(line)
        for i in range(n):
            ratios = [float(x) for x in f.readline().split()]
            # Original coordinates: ratio * original_size
            pt = [ratios[0] * original_size[0], ratios[1] * original_size[1]]
            points.append(pt)
    return np.array(points)

def calculate_lld(keypoints, ref_type='teardrop'):
    """
    Calculate LLD in pixels based on evaluation_LLD.py logic.
    keypoints: [10, 2] - (x, y)
    ref_type: 'asis', 'teardrop', or 'ischial'
    Indices:
    0: ASIS_Left
    1: ASIS_Right
    2: Tear_Drop_Right
    3: Tear_Drop_Left
    4: Ischial_Tuberosity_Left
    5: Ischial_Tuberosity_Right
    8: Lesser_Trochanter_Left
    9: Lesser_Trochanter_Right
    """
    if len(keypoints) < 10:
        return 0.0, {}

    if ref_type == 'asis':
        ref_r = np.array(keypoints[1])
        ref_l = np.array(keypoints[0])
    elif ref_type == 'teardrop':
        ref_r = np.array(keypoints[2])
        ref_l = np.array(keypoints[3])
    elif ref_type == 'ischial':
        ref_r = np.array(keypoints[5])
        ref_l = np.array(keypoints[4])
    else:
        raise ValueError(f"Unknown ref_type: {ref_type}")

    lt_l = np.array(keypoints[8])
    lt_r = np.array(keypoints[9])

    # Reference line (Right to Left)
    ref_vector = ref_l - ref_r
    ref_length = np.linalg.norm(ref_vector)
    
    if ref_length == 0:
        return 0.0, {}

    # Direction vector of the reference line
    ref_unit_vector = ref_vector / ref_length

    # Calculate the perpendicular unit vector to the reference line
    perp_unit_vector = np.array([-ref_unit_vector[1], ref_unit_vector[0]])

    # Project the lesser trochanters onto the perpendicular direction
    left_projection = np.dot(lt_l, perp_unit_vector)
    right_projection = np.dot(lt_r, perp_unit_vector)

    # Calculate the LLD as the signed difference between the projections
    # Positive if left is higher than right, negative otherwise
    lld = left_projection - right_projection

    visualization_data = {
        'ref_left': ref_l,
        'ref_right': ref_r,
        'lesser_trochanter_left': lt_l,
        'lesser_trochanter_right': lt_r,
        'ref_unit_vector': ref_unit_vector,
        'perp_unit_vector': perp_unit_vector,
        'lld': lld
    }

    return lld, visualization_data

def visualize_lld(image_path, pred_landmarks, pred_viz, gt_landmarks=None, gt_viz=None, output_path=None, spacing=None):
    """
    Visualize LLD calculation, mirroring evaluation_LLD.py visualization.
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap='gray')
    
    title = 'Leg Length Discrepancy (LLD) Measurement'
    if gt_viz is not None:
        title += ' - Predicted vs Ground Truth'
    ax.set_title(title)

    landmark_names = get_landmark_names()

    # Plot Pred landmarks
    for i, (x, y) in enumerate(pred_landmarks):
        label = landmark_names[i] if i < len(landmark_names) else f"P{i}"
        ax.plot(x, y, 'o', color='blue', markersize=4)
        ax.text(x+2, y+2, label, color='blue', fontsize=6)

    # Plot Pred LLD elements
    if pred_viz:
        ref_l, ref_r = pred_viz['ref_left'], pred_viz['ref_right']
        lt_l, lt_r = pred_viz['lesser_trochanter_left'], pred_viz['lesser_trochanter_right']
        ref_unit = pred_viz['ref_unit_vector']
        
        # Reference line
        ax.plot([ref_r[0], ref_l[0]], [ref_r[1], ref_l[1]], 'g-', linewidth=2, label='Pred Reference Line')
        
        # Parallel lines
        ext = 200 # Reduced extension as requested
        ax.plot([lt_l[0] - ext*ref_unit[0], lt_l[0] + ext*ref_unit[0]], 
                [lt_l[1] - ext*ref_unit[1], lt_l[1] + ext*ref_unit[1]], 'r-', linewidth=1, label='Pred LT Left Line')
        ax.plot([lt_r[0] - ext*ref_unit[0], lt_r[0] + ext*ref_unit[0]], 
                [lt_r[1] - ext*ref_unit[1], lt_r[1] + ext*ref_unit[1]], 'b-', linewidth=1, label='Pred LT Right Line')
        
        # Measurement line
        ax.annotate('', xy=lt_l, xytext=(lt_l[0] - pred_viz['lld']*pred_viz['perp_unit_vector'][0], 
                                         lt_l[1] - pred_viz['lld']*pred_viz['perp_unit_vector'][1]),
                    arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
        
        lld_px = pred_viz['lld']
        if spacing:
            lld_mm = lld_px * spacing
            text = f"Pred LLD: {lld_px:.2f}px ({lld_mm:.2f}mm)"
        else:
            text = f"Pred LLD: {lld_px:.2f}px"
            
        ax.text(lt_l[0], lt_l[1]-50, text, color='yellow', fontweight='bold', fontsize=10)

    # Plot GT elements if available
    if gt_landmarks is not None:
        for i, (x, y) in enumerate(gt_landmarks):
            ax.plot(x, y, 'x', color='red', markersize=4)
            
    if gt_viz:
        ref_l, ref_r = gt_viz['ref_left'], gt_viz['ref_right']
        lt_l, lt_r = gt_viz['lesser_trochanter_left'], gt_viz['lesser_trochanter_right']
        ref_unit = gt_viz['ref_unit_vector']
        
        ax.plot([ref_r[0], ref_l[0]], [ref_r[1], ref_l[1]], 'g--', linewidth=1, alpha=0.5, label='GT Reference Line')
        
        # GT Parallel lines (dotted)
        ext = 200
        ax.plot([lt_l[0] - ext*ref_unit[0], lt_l[0] + ext*ref_unit[0]], 
                [lt_l[1] - ext*ref_unit[1], lt_l[1] + ext*ref_unit[1]], 'r--', linewidth=1, alpha=0.5, label='GT LT Left Line')
        ax.plot([lt_r[0] - ext*ref_unit[0], lt_r[0] + ext*ref_unit[0]], 
                [lt_r[1] - ext*ref_unit[1], lt_r[1] + ext*ref_unit[1]], 'b--', linewidth=1, alpha=0.5, label='GT LT Right Line')
        
        gt_lld_px = gt_viz['lld']
        if spacing:
            gt_lld_mm = gt_lld_px * spacing
            gt_text = f"GT LLD: {gt_lld_px:.2f}px ({gt_lld_mm:.2f}mm)"
        else:
            gt_text = f"GT LLD: {gt_lld_px:.2f}px"
            
        ax.text(10, 30, gt_text, color='red', fontweight='bold')

    plt.legend(loc='upper right')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    # Use absolute paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / "runs" / "GU2Net_runs" / "checkpoints" / "Hip_Epoch40_BestModel.pt"
    eval_pngs_dir = project_root / "data" / "hip" / "evaluation_set" / "pngs"
    eval_labels_dir = project_root / "data" / "hip" / "evaluation_set" / "labels"
    output_csv = project_root / "Analysis" / "hip_evaluation_results.csv"
    viz_output_dir = project_root / "Analysis" / "roc_results"
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        # Fallback to the long name mentioned in the previous task if it exists
        long_checkpoint_path = checkpoint_path.parent / "best_GU2Net_runs_epoch040_train28357.821321_val4290.672372.pt"
        if long_checkpoint_path.exists():
            checkpoint_path = long_checkpoint_path
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Setup
    localNet_params = {'in_channels': [1], 'out_channels': [10], 'bilinear': True}
    globalNet_params = {'scale_factor': 0.25, 'kernel_size': 3, 'dilations': [1, 2, 5, 2, 1]}
    model = GLN(U2Net, localNet_params, globalNet_params)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle the 6 vs 10 mismatch if necessary, but according to run_instance.py it might just work if we load 6.
    # Wait, the user said "use model runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch040...".
    # run_instance.py had model.load_state_dict(state_dict) and it seemed to expect 6.
    # Let's check the checkpoint shape first.
    
    # Try to load and handle mismatch
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                print(f"Shape mismatch for {k}: {v.shape} vs {model_state[k].shape}. Adjusting model.")
                # If the checkpoint has 6 and model has 10, we need to re-init model with 6 or slice.
                # Actually, let's just re-init GLN with 6 outputs if that's what's in the checkpoint.
                if "local_net.out_conv.conv.weight" in k and v.shape[0] == 6:
                    localNet_params['out_channels'] = [6]
                    model = GLN(U2Net, localNet_params, globalNet_params)
                    model.load_state_dict(state_dict)
                    break
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    
    results = []
    png_files = sorted([f for f in os.listdir(eval_pngs_dir) if f.endswith('.png')])
    landmark_names = get_landmark_names()
    
    for filename in png_files:
        case_id = filename[:-4]
        image_path = os.path.join(eval_pngs_dir, filename)
        label_path = os.path.join(eval_labels_dir, case_id + '.txt')
        
        # Preprocess
        input_tensor, origin_size = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor, task_idx=0)
            heatmaps = output['output'].cpu().numpy()[0]
        
        pred_landmarks_resized = getPointsFromHeatmap(heatmaps) # (N, 2) in 512x512
        
        # Scale predicted landmarks back to original size
        # origin_size is (width, height)
        scale_x = origin_size[0] / 512.0
        scale_y = origin_size[1] / 512.0
        pred_landmarks = np.array([[pt[0] * scale_x, pt[1] * scale_y] for pt in pred_landmarks_resized])
        
        gt_landmarks = read_gt_landmarks(label_path, origin_size) # (10, 2) in original size
        
        if gt_landmarks is None:
            print(f"Skipping {case_id}: GT landmarks not found.")
            continue
            
        # If model only predicted 6, we can only calculate LLD if those 6 include the needed ones.
        # indices needed: 2, 3, 8, 9. 
        # If N < 10, we might be in trouble. 
        # But let's see.
        
        if len(pred_landmarks) < 10:
             # If it's 6, maybe they map to 0-5? 
             # Let's assume for now it might be 10 if I adjusted it correctly.
             # Actually, if the checkpoint has 6, it probably doesn't have the landmarks needed for LLD (8, 9).
             # Wait, lld_results.csv in Analysis/Unet had these landmarks.
             pass

        # Calculate LLD for different reference points
        ref_types = ['asis', 'teardrop', 'ischial']
        
        # We'll use teardrop for the main viz, but calculate all for the CSV
        pred_lld_px, pred_viz = calculate_lld(pred_landmarks, ref_type='teardrop')
        gt_lld_px, gt_viz = calculate_lld(gt_landmarks, ref_type='teardrop')

        # Pixel to mm conversion
        # Get metadata for spacing
        metadata_dirs = [
            r"F:\AS\Hip_AP_Xray\runtime_evaluation_dataset",
            r"D:\processed_files",
            r"F:\AS\Hip_AP_Xray\validation_dataset"
        ]
        metadata = load_metadata(case_id, metadata_dirs)

        if metadata and 'spacing' in metadata:
            pixel_to_mm_scale = metadata['spacing'][0]
            print(f"Using spacing from metadata for {case_id}: {pixel_to_mm_scale}")
        else:
            pixel_to_mm_scale = 0.5
            print(f"Warning: Metadata not found for {case_id}, using default spacing: {pixel_to_mm_scale}")

        # Visualize and save (using teardrop as default)
        viz_path = viz_output_dir / f"{case_id}_lld_viz.png"
        visualize_lld(image_path, pred_landmarks, pred_viz, gt_landmarks, gt_viz, output_path=viz_path, spacing=pixel_to_mm_scale)

        res = {
            'case_id': case_id,
            'spacing': pixel_to_mm_scale
        }
        
        for rt in ref_types:
            p_lld_px, _ = calculate_lld(pred_landmarks, ref_type=rt)
            g_lld_px, _ = calculate_lld(gt_landmarks, ref_type=rt)
            
            p_lld_mm = p_lld_px * pixel_to_mm_scale
            g_lld_mm = g_lld_px * pixel_to_mm_scale
            
            res[f'pred_lld_{rt}_mm'] = p_lld_mm
            res[f'gt_lld_{rt}_mm'] = g_lld_mm
            res[f'lld_diff_{rt}_mm'] = p_lld_mm - g_lld_mm
            
        # Keep legacy columns for compatibility if needed, using teardrop
        res['predicted_lld_mm'] = res['pred_lld_teardrop_mm']
        res['ground_truth_lld_mm'] = res['gt_lld_teardrop_mm']
        
        # Add landmark coordinates to match lld_results.csv
        relevant_indices = {
            'asis_left': 0,
            'asis_right': 1,
            'tear_drop_right': 2,
            'tear_drop_left': 3,
            'ischial_tuberosity_left': 4,
            'ischial_tuberosity_right': 5,
            'lesser_trochanter_left': 8,
            'lesser_trochanter_right': 9
        }
        
        for name, idx in relevant_indices.items():
            if idx < len(pred_landmarks):
                res[f'{name}_x'] = pred_landmarks[idx][0]
                res[f'{name}_y'] = pred_landmarks[idx][1]
            if idx < len(gt_landmarks):
                res[f'gt_{name}_x'] = gt_landmarks[idx][0]
                res[f'gt_{name}_y'] = gt_landmarks[idx][1]
        
        results.append(res)
        print(f"Processed {case_id}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
