import os
from PIL import Image
import sys
import os
sys.path.append('/home/waleed/Documents/deeplense25')

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import MAEViT, print_trainable_parameters
import torchvision.models as models
from Dataset import NPYDataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from specific_test_06.ran.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches


# Show samples
class_namesB = ["LR", "HR"]
dataset_pathB = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset"
# show_sample_images(dataset_pathB, class_namesB)
class_namesA = ["cdm", "no_sub", "axion"]
dataset_pathA = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset"
show_sample_images(dataset_pathA, class_namesA)

# Load an example image (grayscale from .npy)
image_path = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/no_sub/no_sub_sim_100008917700138988486624773794898508448.npy"
image_path = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/LR/sample100.npy"
image_path2 = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/HR/sample100.npy"
patch_size = 15
input_dim = 225
num_patches = 10
image_array = np.load(image_path)  # Load .npy file
image_tensor = torch.tensor(image_array, dtype=torch.float32).squeeze(0)# .squeeze(0)  # Convert to tensor
print(image_tensor.shape)
image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  # Normalize

# Convert to PIL image (for transformations)
image = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8))  # Convert back to uint8

# Process image into patches
img_patches = image_to_patches(image, patch_size)

# Visualize the patches
# plt.figure(figsize=(6, 6))
# for i in range(15):
#     for j in range(15):
#         idx = i * 15 + j
#         patch = img_patches[0, idx, :].view(1, 10, 10).squeeze(0).numpy()  # Convert back to 2D
#         plt.subplot(15, 15, idx + 1)
#         plt.imshow(patch, cmap="gray")  # Display as grayscale
#         plt.axis('off')
# plt.show()

visible_patches, masked_indices, visible_indices = random_masking(img_patches, mask_ratio=0.75)

# Visualize the remaining patches
visualize_patches(visible_patches, visible_indices, original_size=(150, 150), patch_size=patch_size, title="Sample masked HR image (15, 15)")

# Load Pretrained ViT Model
model = MAEViT(input_dim=input_dim, num_patches=225)
print(img_patches.shape, visible_patches.shape, masked_indices.shape, visible_indices.shape)

output, mask = model(img_patches)
print(mask.shape)

masked_output = torch.gather(output, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))
masked_patches = torch.gather(img_patches, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))

# print(masked_output.shape)
# print(masked_patches.shape)
# print(masked_output.shape, mask.shape)
# print(masked_output.dtype, mask.dtype)
# print(masked_output.min(), mask.min())
# print(masked_output.max(), mask.max())

# visualize_patches(masked_patches, mask, original_size=(224, 224), patch_size=16, title="Masked patches")

path_a_no_sub = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/no_sub"
path_a_cdm = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/cdm"
path_a_axion = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/axion"
"""
    # Get total number of samples
    all_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    num_samples = len(all_files)
    
    # Create indices and split 90:10
    indices = np.arange(num_samples)
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

train_data = NPYDataset(path_a)"""