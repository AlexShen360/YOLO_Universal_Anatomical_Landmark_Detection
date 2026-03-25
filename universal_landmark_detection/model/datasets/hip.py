import os
import json
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from ..utils import gaussianHeatmap, transformer


class Hip(data.Dataset):

    def __init__(self, prefix, phase, transform_params=dict(), sigma=5, num_landmark=10, size=[512, 512], use_background_channel=False):

        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel
        self.prefix = prefix

        # Get all .npy files (images)
        files = []
        for f in os.listdir(prefix):
            if f.endswith('.npy'):
                files.append(f[:-4])  # Remove .npy extension
        files = sorted(files)
        
        n = len(files)
        train_num = int(n * 0.7)
        val_num = int(n * 0.15)
        test_num = n - train_num - val_num
        
        if phase == 'train':
            self.indexes = files[:train_num]
        elif phase == 'validate':
            self.indexes = files[train_num:train_num + val_num]
        elif phase == 'test':
            self.indexes = files[train_num + val_num:]
        else:
            raise Exception(f"Unknown phase: {phase}")
            
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, origin_size = self.readImage(os.path.join(self.prefix, name + '.npy'))
        
        points = self.readLandmark(name, origin_size)
        li = [self.genHeatmap(point, self.size) for point in points]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm > 1] = 1
            li.append(1 - sm)
        gt = np.array(li)
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        csv_path = os.path.join(self.prefix, name + '_landmarks.csv')
        json_path = os.path.join(self.prefix, name + '.json')
        
        # Read CSV landmarks
        df = pd.read_csv(csv_path, index_col=0)
        
        # Read JSON metadata to get original and resized dimensions
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        original_dims = metadata['original_dimensions']  # [width, height]
        resized_dims = metadata['resized_dimensions']    # [width, height]
        
        points = []
        for _, row in df.iterrows():
            x, y = row['X'], row['Y']
            
            # Convert from resized coordinates to target size coordinates
            # First convert to ratios based on resized dimensions
            x_ratio = x / resized_dims[0]
            y_ratio = y / resized_dims[1]
            
            # Then convert to target size
            pt = (int(x_ratio * self.size[0]), int(y_ratio * self.size[1]))
            points.append(pt)
            
        return points

    def readImage(self, path):
        '''Read image from .npy file and return a numpy.ndarray in shape of cxwxh
        '''
        # Load .npy file
        arr = np.load(path, allow_pickle=True)
        origin_size = arr.shape[:2]  # (height, width)
        
        # Resize to target size
        # Convert to PIL Image for resizing
        if arr.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        img = Image.fromarray(arr)
        img = img.resize(self.size)
        arr = np.array(img)
        
        # Convert to channel x width x height: 1 x width x height
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float64)
        
        # Normalize
        for i in range(arr.shape[0]):
            arr[i] = (arr[i] - arr[i].mean()) / (arr[i].std() + 1e-20)
            
        return arr, origin_size

    def visualize_sample(self, index, save_dir='visualizations'):
        """Visualize a sample with landmarks and save to file"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        name = self.indexes[index]
        
        # Load original image
        img_path = os.path.join(self.prefix, name + '.npy')
        arr = np.load(img_path)
        
        # Load landmarks
        csv_path = os.path.join(self.prefix, name + '_landmarks.csv')
        df = pd.read_csv(csv_path, index_col=0)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(arr, cmap='gray')
        
        # Plot landmarks
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        for i, (landmark_name, row) in enumerate(df.iterrows()):
            x, y = row['X'], row['Y']
            plt.plot(x, y, 'o', color=colors[i], markersize=8, label=landmark_name)
            plt.text(x + 10, y + 10, landmark_name, color=colors[i], fontsize=8)
        
        plt.title(f'Hip X-ray with Landmarks - {name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(save_dir, f'{name}_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {save_path}")
        return save_path