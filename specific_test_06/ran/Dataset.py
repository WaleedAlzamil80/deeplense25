from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.npy_files = npy_dir
        self.transform = transform

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        img = np.load(self.npy_files[idx])
        img = Image.fromarray(np.uint8(img * 255))

        if self.transform:
            img = self.transform(img)

        return img

class NPYClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of file paths to .npy files.
            labels (list): Corresponding labels for classification.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]

        img = np.load(self.file_paths[idx], allow_pickle=True)
        if label == 1:
            img = img[0]
        img = Image.fromarray(np.uint8(img * 255))

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


class NPYSuperResolutionDataset(Dataset):
    def __init__(self, samples_list, base_dir, transform_hr=None, transform_lr=None):
        self.base_dir = base_dir
        self.file_names = samples_list
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Load HR and LR images
        hr_img = np.load(os.path.join(self.base_dir, "HR", file_name))
        lr_img = np.load(os.path.join(self.base_dir, "LR", file_name))

        # Convert to PIL Image
        hr_img = Image.fromarray(np.uint8(hr_img.squeeze() * 255))
        lr_img = Image.fromarray(np.uint8(lr_img.squeeze() * 255))

        # Apply transformations
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)

        return lr_img, hr_img
