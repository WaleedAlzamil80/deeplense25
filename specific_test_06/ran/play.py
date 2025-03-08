import os
from PIL import Image
import sys
import os
sys.path.append('/home/waleed/Documents/deeplense25')

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from model import MAEViT, print_trainable_parameters
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from specific_test_06.ran.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches


# Show samples
class_namesB = ["LR", "HR"]
dataset_pathB = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset"
show_sample_images(dataset_pathB, class_namesB)
class_namesA = ["cdm", "no_sub", "axion"]
dataset_pathA = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset"
show_sample_images(dataset_pathA, class_namesA)

# Load an example image (grayscale from .npy)
image_path = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/no_sub/no_sub_sim_100008917700138988486624773794898508448.npy"
image_array = np.load(image_path)  # Load .npy file
image_tensor = torch.tensor(image_array, dtype=torch.float32)  # Convert to tensor
image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  # Normalize

# Convert to PIL image (for transformations)
image = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8))  # Convert back to uint8

# Process image into patches
img_patches = image_to_patches(image)

# Visualize the patches
plt.figure(figsize=(6, 6))
for i in range(14):
    for j in range(14):
        idx = i * 14 + j
        patch = img_patches[0, idx, :].view(1, 16, 16).squeeze(0).numpy()  # Convert back to 2D
        plt.subplot(14, 14, idx + 1)
        plt.imshow(patch, cmap="gray")  # Display as grayscale
        plt.axis('off')
plt.show()

visible_patches, masked_indices, visible_indices = random_masking(img_patches, mask_ratio=0.75)
print(visible_patches.shape)
# Visualize the remaining patches
visualize_patches(visible_patches, visible_indices, original_size=(224, 224), patch_size=16)

# Load Pretrained ViT Model
