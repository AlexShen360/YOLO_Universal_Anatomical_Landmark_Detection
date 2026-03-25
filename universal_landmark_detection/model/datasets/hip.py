import os
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data

from ..utils import gaussianHeatmap, transformer


class Hip(data.Dataset):

    def __init__(self, prefix, phase, transform_params=dict(), sigma=5, num_landmark=10, size=[512, 512], use_background_channel=False):

        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel

        # Handle path resolution for different working directories
        if os.path.exists(os.path.join(prefix, 'pngs')):
            self.pth_Image = os.path.join(prefix, 'pngs')
            self.pth_Label = os.path.join(prefix, 'labels')
        elif os.path.exists(os.path.join('..', prefix, 'pngs')):
            self.pth_Image = os.path.join('..', prefix, 'pngs')
            self.pth_Label = os.path.join('..', prefix, 'labels')
        else:
            # Try absolute path resolution
            abs_prefix = os.path.abspath(prefix)
            if os.path.exists(os.path.join(abs_prefix, 'pngs')):
                self.pth_Image = os.path.join(abs_prefix, 'pngs')
                self.pth_Label = os.path.join(abs_prefix, 'labels')
            else:
                raise FileNotFoundError(f"Cannot find hip data directory. Tried: {prefix}, ../{prefix}, {abs_prefix}")

        # Get all .png files (images)
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

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

        img, origin_size = self.readImage(os.path.join(self.pth_Image, name + '.png'))

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
        path = os.path.join(self.pth_Label, name + '.txt')
        points = []
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                ratios = [float(i) for i in f.readline().split()]
                pt = tuple([round(r*sz) for r, sz in zip(ratios, self.size)])
                points.append(pt)
        return points

    def readImage(self, path):
        '''Read image from path and return a numpy.ndarray in shape of cxwxh
        '''
        img = Image.open(path)
        origin_size = img.size

        # resize, width x height,  channel=1
        img = img.resize(self.size)
        arr = np.array(img)
        # channel x width x height: 1 x width x height
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float64)
        # conveting to float is important, otherwise big bug occurs
        for i in range(arr.shape[0]):
            arr[i] = (arr[i]-arr[i].mean())/(arr[i].std()+1e-20)
        return arr, origin_size
