import sys
import os
sys.path.append('/home/waleed/Documents/deeplense25/specific_test_06')
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYDataset
from utils.helpful import image_to_patches
from models.mae import MAEViT

train_transforms = transforms.Compose([
    # transforms.CenterCrop(100),
    transforms.Resize(150, Image.LANCZOS),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
        transforms.Resize(150, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset_root = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/"

axion_files = sorted(glob(os.path.join(dataset_root, "axion", "*.npy")))
# [f for f in os.listdir(os.path.join(dataset_root, "axion", "*.npy"))]

no_sub_files = sorted(glob(os.path.join(dataset_root, "no_sub", "*.npy")))
cdm_files = sorted(glob(os.path.join(dataset_root, "cdm", "*.npy")))

all_files = no_sub_files + axion_files + cdm_files
labels = [0] * len(no_sub_files) + [1] * len(axion_files) + [2] * len(cdm_files)

# First split: 90% train, 10% val (stratified)
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, labels, test_size=0.1, stratify=labels, random_state=42
)

# Filter out only `no_sub` samples from training set
no_sub_train_files = [f for f, l in zip(train_files, train_labels) if l == 0]
no_sub_val_files = [f for f, l in zip(val_files, val_labels) if l == 0]

# Train MAE only on no_sub_train_files
batch_size=64
train_dataset = NPYDataset(no_sub_train_files, train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation set for MAE (later used in classification also)
val_dataset = NPYDataset(no_sub_val_files, val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MAEViT(base="tiny", embed_dim=192, input_dim=input_dim, num_patches=num_patches).to(device)
model = nn.DataParallel(model)
base_model = "/home/waleed/Documents/deeplense25/specific_test_06/models/checkpoints/mae.pth"
state_dict = torch.load(base_model, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

for images in val_loader:
    images = images.to(device)
    # plt.imsave("original_image.png", images[0].cpu().detach().view(150, 150).numpy(), cmap="gray")

    images = image_to_patches(images, 10)
    output, mask = model(images)