import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add the project directory to sys.path to import modules
sys.path.append(os.path.abspath('.'))

from universal_landmark_detection.model.datasets.hip import Hip
from universal_landmark_detection.model.datasets.chest import Chest

def visualize_sample(dataset, index, title, ax):
    sample = dataset[index]
    img = sample['input'].numpy()[0] # (1, W, H) -> (W, H)
    gt = sample['gt'].numpy() # (N, W, H)
    
    # Transpose back to (H, W) for plotting
    img = np.transpose(img, (1, 0))
    
    # Get landmarks from heatmaps
    landmarks = []
    for i in range(gt.shape[0]):
        # Skip background channel if present
        if i >= dataset.num_landmark:
            break
        heatmap = gt[i]
        # Transpose back to (H, W) to match image
        heatmap = np.transpose(heatmap, (1, 0))
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        landmarks.append((x, y))
    
    # Landmark names for Hip
    hip_names = [
        "ASIS_L", "ASIS_R", "TD_R", "TD_L", "IT_L", "IT_R", "FHC_L", "FHC_R", "LT_L", "LT_R"
    ]
    
    ax.imshow(img, cmap='gray')
    for i, (x, y) in enumerate(landmarks):
        ax.scatter(x, y, c='red', s=20)
        label = hip_names[i] if "Hip" in title and i < len(hip_names) else str(i)
        ax.text(x, y, label, color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.5))
    ax.set_title(f"{title}: {sample['name']}")
    ax.axis('off')

def main():
    # Parameters from config.yaml (mostly)
    hip_params = {
        'prefix': 'data/hip',
        'phase': 'test',
        'sigma': 5,
        'num_landmark': 10,
        'size': [512, 512]
    }
    
    chest_params = {
        'prefix': 'data/chest',
        'phase': 'test',
        'sigma': 5,
        'num_landmark': 6,
        'size': [512, 512]
    }
    
    hip_dataset = Hip(**hip_params)
    chest_dataset = Chest(**chest_params)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    print("Visualizing Hip sample...")
    visualize_sample(hip_dataset, 0, "Hip Dataset", axes[0])
    
    print("Visualizing Chest sample...")
    visualize_sample(chest_dataset, 0, "Chest Dataset", axes[1])
    
    output_path = "dataset_alignment_check.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
