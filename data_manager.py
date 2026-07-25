import glob
import cv2
import random
import numpy as np
import pickle
import os

from torch.utils import data
import torch


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
        base_dir = config.datasets_dir

        self.input_dir = 'cloudy_image' if os.path.exists(os.path.join(base_dir, 'cloudy_image')) else 'cloud'
        self.target_dir = 'ground_truth' if os.path.exists(os.path.join(base_dir, 'ground_truth')) else 'label'
        self.mask_dir = 'mask' if os.path.exists(os.path.join(base_dir, 'mask')) else None
        
        # Save to config for verification printout
        config.detected_input_dir = self.input_dir
        config.detected_target_dir = self.target_dir

        train_list_file = os.path.join(base_dir, 'train_list.txt')
        test_list_file = os.path.join(base_dir, 'test_list.txt')
        
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = [f for f in os.listdir(os.path.join(base_dir, self.input_dir)) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            files.sort()
            n_train = int(0.8 * len(files))
            train_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(train_list_file, np.array(train_list), fmt='%s')
            np.savetxt(test_list_file, np.array(test_list), fmt='%s')

        self.imlist = np.loadtxt(train_list_file, str)
        if self.imlist.ndim == 0:
            self.imlist = np.array([self.imlist])

        # Also load test list just to store its size for verification printout
        if os.path.exists(test_list_file):
            test_imlist = np.loadtxt(test_list_file, str)
            if test_imlist.ndim == 0: test_imlist = np.array([test_imlist])
            config.val_size = len(test_imlist)
        else:
            config.val_size = 0
        config.train_size = len(self.imlist)

    def __getitem__(self, index):
        
        t = cv2.imread(os.path.join(self.config.datasets_dir, self.target_dir, str(self.imlist[index])), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, self.input_dir, str(self.imlist[index])), 1).astype(np.float32)

        if self.mask_dir:
            M_path = os.path.join(self.config.datasets_dir, self.mask_dir, str(self.imlist[index]))
            if os.path.exists(M_path):
                M_img = cv2.imread(M_path, 0)
                if M_img is not None:
                    M = (M_img / 255.0).astype(np.float32)
                else:
                    M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
            else:
                M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        else:
            M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
            
        # Convert BGR to RGB
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        x = x.astype(np.float32) / 127.5 - 1.0
        t = t.astype(np.float32) / 127.5 - 1.0
        
        if hasattr(self.config, 'image_size'):
            size = (self.config.image_size, self.config.image_size)
            x = cv2.resize(x, size)
            t = cv2.resize(t, size)
            M = cv2.resize(M, size)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)
        if t.ndim == 2:
            t = np.expand_dims(t, axis=2)
        if M.ndim == 2:
            M = np.expand_dims(M, axis=2)

        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        M = M.transpose(2, 0, 1)

        # Duplicate channel 0 as a placeholder NIR band to make it 4 channels
        x = np.concatenate([x, x[0:1, :, :]], axis=0)
        t = np.concatenate([t, t[0:1, :, :]], axis=0)

        # Dummy SAR data (2 bands)
        sar = np.zeros((2, x.shape[1], x.shape[2]), dtype=np.float32)

        return {
            'input': torch.from_numpy(x).float(),
            'target': torch.from_numpy(t).float(),
            'sar': torch.from_numpy(sar).float(),
            'mask': torch.from_numpy(M).float()
        }

    def __len__(self):
        return len(self.imlist)


class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        self.test_dir = test_dir
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.test_files = os.listdir(os.path.join(test_dir, 'cloudy'))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        
        x = cv2.imread(os.path.join(self.test_dir, 'cloudy', filename), 1).astype(np.float32)
        # Convert BGR to RGB
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        x = x.astype(np.float32) / 127.5 - 1.0
        x = x.transpose(2, 0, 1)
        
        # Duplicate channel 0 as a placeholder NIR band to make it 4 channels
        x = np.concatenate([x, x[0:1, :, :]], axis=0)
        
        # Dummy SAR data (2 bands)
        sar = np.zeros((2, x.shape[1], x.shape[2]), dtype=np.float32)

        return {
            'input': torch.from_numpy(x).float(),
            'sar': torch.from_numpy(sar).float(),
            'filename': filename
        }

    def __len__(self):
        return len(self.test_files)
