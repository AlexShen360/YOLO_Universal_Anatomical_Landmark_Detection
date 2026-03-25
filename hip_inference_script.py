import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.append('.')

from universal_landmark_detection.model.utils import getPointsFromHeatmap, mkdir
from universal_landmark_detection.model.networks import get_net
from universal_landmark_detection.model.utils.yamlConfig import get_config

def load_model_and_config(checkpoint_path, config_path='universal_landmark_detection/config.yaml'):
    """Load model from checkpoint and configuration"""

    # Load configuration
    opts = get_config(config_path)

    # Setup model parameters for multi-dataset model
    # Override model type to match the checkpoint (GLN model was used for training)
    modelname = 'gln'  # The checkpoint was trained with GLN model, not unet2d

    # Get dataset configuration - match exactly what was trained (chest only)
    dataset = opts.dataset
    trained_datasets = ['chest']  # The checkpoint was trained only on chest dataset

    # Setup channel parameters to match the training configuration exactly
    channel_params = {'in_channels': [], 'out_channels': []}
    for name in trained_datasets:
        if name in dataset:
            dataset_config = dataset[name]
            channel_params['in_channels'].append(1)  # All datasets use single channel
            channel_params['out_channels'].append(dataset_config['num_landmark'])

    # Create model with exact same architecture as training
    if modelname.startswith('gln'):
        # For GLN models
        model_opts = opts[modelname] if modelname in opts else {}
        localNet = model_opts['localNet']
        # Convert EasyDict to regular dict and copy
        globalNet_params = dict(model_opts['globalNet_params'])
        globalNet_params.update(channel_params)
        localNet_params = dict(channel_params)
        model = get_net(modelname)(get_net(localNet), localNet_params, globalNet_params)
    else:
        # For other models like unet2d
        net_params = opts[modelname] if modelname in opts else {}
        net_params.update(channel_params)
        model = get_net(modelname)(**net_params)

    # Load checkpoint
    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded from {checkpoint_path}")
    print(f"Device: {device}")

    return model, opts, device

def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess image for inference"""

    # Load .npy file
    arr = np.load(image_path, allow_pickle=True)
    original_size = arr.shape[:2]  # (height, width)

    # Normalize to 0-255 range if needed
    if arr.dtype != np.uint8:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Resize to target size
    img = Image.fromarray(arr)
    img_resized = img.resize(target_size)
    arr_resized = np.array(img_resized)

    # Convert to channel x width x height: 1 x width x height
    if arr_resized.ndim == 3:
        arr_resized = arr_resized[..., 0]
    arr_processed = np.expand_dims(np.transpose(arr_resized, (1, 0)), 0).astype(np.float64)

    # Normalize
    for i in range(arr_processed.shape[0]):
        arr_processed[i] = (arr_processed[i] - arr_processed[i].mean()) / (arr_processed[i].std() + 1e-20)

    return arr_processed, arr, original_size

def get_landmark_names(csv_path):
    """Get landmark names from CSV file"""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return list(df.index)
    else:
        # Default hip landmark names if CSV not available
        return [f'Landmark_{i+1}' for i in range(10)]

def overlay_landmarks_on_image(image_array, landmarks, landmark_names, output_path):
    """Overlay landmarks on image and save"""

    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        # Normalize to 0-255
        img_min, img_max = image_array.min(), image_array.max()
        if img_max > img_min:
            image_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    # Convert to RGB for colored landmarks
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array, image_array, image_array], axis=-1)

    img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img)

    # Try to load font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Define colors for landmarks
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']

    # Draw landmarks
    for i, (landmark, name) in enumerate(zip(landmarks, landmark_names)):
        x, y = landmark
        color = colors[i % len(colors)]

        # Draw circle for landmark
        radius = 5
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='white', width=2)

        # Draw landmark name
        text_x, text_y = x + 10, y - 10
        draw.text((text_x, text_y), name, fill=color, font=font)

    # Save image
    img.save(output_path)
    print(f"Saved annotated image: {output_path}")

def run_inference(model, device, input_path, output_path):
    """Run inference on images in input_path and save results to output_path"""

    # Create output directory
    mkdir(output_path)

    # Get all .npy files in input directory
    image_files = [f for f in os.listdir(input_path) if f.endswith('.npy') and not f.endswith('_landmarks.csv')]

    if not image_files:
        print(f"No .npy image files found in {input_path}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_name = image_file[:-4]  # Remove .npy extension
        image_path = os.path.join(input_path, image_file)

        try:
            # Load and preprocess image
            processed_img, original_img, original_size = load_and_preprocess_image(image_path)

            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_img).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                output = model(input_tensor, 0)  # task_idx=0 since model was trained only on chest dataset
                heatmaps = output['output'].detach().cpu().numpy()[0]  # Remove batch dimension

            # Extract landmarks from heatmaps
            landmarks = getPointsFromHeatmap(heatmaps)

            # Get landmark names
            csv_path = os.path.join(input_path, image_name + '_landmarks.csv')
            landmark_names = get_landmark_names(csv_path)

            # Scale landmarks back to original image size
            target_size = (512, 512)
            scaled_landmarks = []
            for landmark in landmarks:
                x_scaled = int(landmark[0] * original_size[1] / target_size[0])  # width scaling
                y_scaled = int(landmark[1] * original_size[0] / target_size[1])  # height scaling
                scaled_landmarks.append((x_scaled, y_scaled))

            # Create output image with overlaid landmarks
            output_image_path = os.path.join(output_path, f"{image_name}_landmarks.png")
            overlay_landmarks_on_image(original_img, scaled_landmarks, landmark_names, output_image_path)

            # Save landmark coordinates
            results_path = os.path.join(output_path, f"{image_name}_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Image: {image_name}\n")
                f.write(f"Original size: {original_size}\n")
                f.write(f"Landmarks:\n")
                for name, (x, y) in zip(landmark_names, scaled_landmarks):
                    f.write(f"{name}: ({x}, {y})\n")

            # Save heatmaps
            heatmaps_path = os.path.join(output_path, f"{image_name}_heatmaps.npy")
            np.save(heatmaps_path, heatmaps)

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    print(f"Inference completed. Results saved to {output_path}")

def main():
    # Configuration
    checkpoint_path = "runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch042_train3791.683512_val965.346590.pt"
    input_path = r"F:\AS\Hip_AP_Xray\runtime_evaluation_dataset"
    output_path = "./evaluation"

    print("Hip X-ray Landmark Detection Inference")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print("=" * 50)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    # Check if input directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input directory not found: {input_path}")
        return

    try:
        # Load model
        model, opts, device = load_model_and_config(checkpoint_path)

        # Run inference
        run_inference(model, device, input_path, output_path)

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
