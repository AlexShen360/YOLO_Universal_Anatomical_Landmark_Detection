import os
import numpy as np
import pandas as pd
from PIL import Image
import glob

def convert_npy_to_png(npy_path, output_path):
    """Convert .npy file to .png format"""
    # Load the numpy array
    data = np.load(npy_path)
    
    # Convert from uint16 to uint8 by normalizing
    # Scale from 0-16383 to 0-255
    data_normalized = (data / data.max() * 255).astype(np.uint8)
    
    # Create PIL image and save as PNG
    img = Image.fromarray(data_normalized)
    img.save(output_path)
    print(f"Converted {npy_path} -> {output_path}")

def convert_csv_to_txt(csv_path, output_path, image_width=1024, image_height=1024):
    """Convert CSV landmarks to TXT format with normalized coordinates"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the coordinates (skip the first column which is landmark names)
    coordinates = []
    for _, row in df.iterrows():
        if pd.notna(row['X']) and pd.notna(row['Y']):
            # Normalize coordinates to 0-1 range
            x_norm = row['X'] / image_width
            y_norm = row['Y'] / image_height
            coordinates.append((x_norm, y_norm))
    
    # Write to TXT file
    with open(output_path, 'w') as f:
        # First line: number of landmarks
        f.write(f"{len(coordinates)}\n")
        # Following lines: normalized coordinates
        for x, y in coordinates:
            f.write(f"{x} {y}\n")
    
    print(f"Converted {csv_path} -> {output_path}")

def main():
    # Source and target directories
    source_dir = r"D:\processed_files"
    target_base_dir = r"C:\GitHub\YOLO_Universal_Anatomical_Landmark_Detection\data\hip"
    
    # Create target directories
    target_pngs_dir = os.path.join(target_base_dir, "pngs")
    target_labels_dir = os.path.join(target_base_dir, "labels")
    
    os.makedirs(target_pngs_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    # Get all .npy files in source directory
    npy_files = glob.glob(os.path.join(source_dir, "*.npy"))
    
    print(f"Found {len(npy_files)} .npy files to convert")
    
    # Convert each file
    for i, npy_file in enumerate(npy_files):
        # Extract the base filename (without extension)
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        
        # Create target filename in CHNCXR format
        target_name = f"HIPAPR_{base_name}_0"
        
        # Convert .npy to .png
        png_output = os.path.join(target_pngs_dir, f"{target_name}.png")
        convert_npy_to_png(npy_file, png_output)
        
        # Convert corresponding .csv to .txt
        csv_file = os.path.join(source_dir, f"{base_name}_landmarks.csv")
        if os.path.exists(csv_file):
            txt_output = os.path.join(target_labels_dir, f"{target_name}.txt")
            convert_csv_to_txt(csv_file, txt_output)
        else:
            print(f"Warning: No landmarks file found for {base_name}")
    
    print(f"\nConversion complete!")
    print(f"Images saved to: {target_pngs_dir}")
    print(f"Labels saved to: {target_labels_dir}")

if __name__ == "__main__":
    main()