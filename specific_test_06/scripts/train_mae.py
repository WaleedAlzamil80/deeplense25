import os
from PIL import Image
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYDataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
from specific_test_06.utils.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches
from models.mae import MAEViT
from utils.helpful import print_trainable_parameters

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
batch_size=256
train_dataset = NPYDataset(no_sub_train_files, train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True)

# Validation set for MAE (later used in classification also)
val_dataset = NPYDataset(no_sub_val_files, val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MAEViT(input_dim=input_dim, num_patches=num_patches).to(device)
print("masked patches: ", int(0.75*225))
print("visible patches: ", num_patches - int(0.75*225))
print_trainable_parameters(model)


# Optimizer & Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-6) #  , weight_decay=2e-4 , weight_decay=1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)
criterion = nn.MSELoss()

# Track metrics
train_losses, val_losses = [], []

# Training Loop
num_epochs = 50
best_val_loss = 10000000.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0


    for images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images = images.to(device)
        plt.imsave("original_image.png", images[0].cpu().detach().view(150, 150).numpy(), cmap="gray")

        optimizer.zero_grad()
        images = image_to_patches(images, 10)
        output, mask = model(images)
        masked_output = torch.gather(output, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))
        masked_patches = torch.gather(images, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))
        loss = criterion(masked_output.view(images.shape[0], -1), masked_patches.view(images.shape[0], -1))

        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation Step
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            plt.imsave("original_image.png", images[0].cpu().detach().view(150, 150).numpy(), cmap="gray")

            optimizer.zero_grad()
            images = image_to_patches(images, 10)
            output, mask = model(images)
            masked_output = torch.gather(output, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))
            masked_patches = torch.gather(images, dim=1, index=mask.unsqueeze(-1).expand(-1, -1, input_dim))
            val_loss += criterion(masked_output.view(images.shape[0], -1), masked_patches.view(images.shape[0], -1)).item()

    val_loss = val_loss / len(val_loader)
    scheduler.step(val_loss) 

    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_vit_MAE0000_model.pth")
        print("Model Saved (Best Validation loss)")

epochs = range(1, num_epochs + 1)
save_dir = "/home/waleed/Documents/deeplense25/specific_test_06/assets"

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig(os.path.join(save_dir, 'MAE_Losses.png'))
plt.show()
