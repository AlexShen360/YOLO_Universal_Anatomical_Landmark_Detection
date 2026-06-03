import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add the project directory to sys.path to import modules
sys.path.append(os.path.abspath('.'))

from universal_landmark_detection.model.networks.u2net import U2Net
from universal_landmark_detection.model.networks.gln import GLN
from universal_landmark_detection.model.utils.kit import getPointsFromHeatmap
import pandas as pd

def get_landmark_names(csv_path, num_landmarks):
    """Get landmark names from CSV file or return defaults"""
    # Provided sequence of landmarks
    provided_names = [
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
    
    if num_landmarks <= len(provided_names):
        return provided_names[:num_landmarks]
    
    # Fallback if more landmarks are predicted than named
    extra_names = [f"L{i+1}" for i in range(len(provided_names), num_landmarks)]
    return provided_names + extra_names

def preprocess_image(path, size=(512, 512)):
    """Follows logic in universal_landmark_detection/model/datasets/hip.py"""
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
        
    return torch.FloatTensor(arr).unsqueeze(0), img_resized, origin_size

def main():
    # Paths
    image_path = "data/hip/evaluation_set/pngs/HIPAPR_13042_0.png"
    # Find the best checkpoint for epoch 70 (as requested in the original issue)
    checkpoint_path = "runs/GU2Net_runs/checkpoints/Hip_Epoch40_BestModel.pt"
    output_dir = "hip_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters from config_train.yaml
    # Note: trained on 'chest' first, then 'hip' probably.
    # config_train.yaml says name_list: ['chest'] but GU2Net_runs suggests hip was also involved.
    # Actually, the task is for hip. Hip has 10 landmarks.
    
    localNet_params = {
        'in_channels': [1],
        'out_channels': [10],
        'bilinear': True
    }
    globalNet_params = {
        'scale_factor': 0.25,
        'kernel_size': 3,
        'dilations': [1, 2, 5, 2, 1]
    }
    
    # Initialize GLN with U2Net as local network
    model = GLN(U2Net, localNet_params, globalNet_params)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # The checkpoint actually has 6 outputs for task 0.
    # We want 10 landmarks for hip.
    # It seems the model in the checkpoint was ONLY configured with 6 outputs.
    # But the task asks for landmark prediction on a HIP image.
    # Maybe the 6 outputs ARE some hip landmarks? Or it was trained on chest but we should use it?
    # Actually, if I look at the error again:
    # "copying a param with shape torch.Size([6, 64, 1, 1]) from checkpoint, the shape in current model is torch.Size([10, 64, 1, 1])"
    # This means the checkpoint DEFINITELY has 6 outputs.
    
    # Let's try to load it with 6 outputs and see what happens.
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Preprocess image
    input_tensor, img_resized, origin_size = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor, task_idx=0)
        heatmaps = output['output'].cpu().numpy()[0] # (6, 512, 512)
    
    # Extract landmarks
    landmarks = getPointsFromHeatmap(heatmaps)
    
    # Get landmark names (try to find a matching csv if possible, otherwise default)
    # The user mentioned an example CSV path, but we'll use a local search or default
    landmark_names = get_landmark_names(None, len(landmarks))
    
    # Visualize
    plt.figure(figsize=(10, 10))
    img_array = np.array(img_resized)
    plt.imshow(img_array, cmap='gray')
    
    # Define colors for landmarks
    colors = plt.get_cmap('tab10')
    
    for i, (x, y) in enumerate(landmarks):
        color = colors(i % 10)
        plt.scatter(x, y, color=color, s=40, edgecolors='white')
        plt.text(x + 5, y - 5, landmark_names[i], color=color, fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        
    plt.title(f"Predicted Landmarks for {os.path.basename(image_path)}")
    plt.axis('off')
    
    save_path = os.path.join(output_dir, f"{os.path.basename(image_path)[:-4]}_with_names.png")
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    main()
