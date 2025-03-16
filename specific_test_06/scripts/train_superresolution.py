import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYSuperResolutionDataset
from utils.helpful import print_trainable_parameters
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
from specific_test_06.utils.helpful import image_to_patches
from models.super_resolution import SuperResolutionViT
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_torch  # Fast SSIM on GPU


class PairedTransform:
    def __init__(self, transform, save_dir="debug_images", debug=False):
        self.transform = transform
        self.save_dir = save_dir
        self.debug = debug
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    def __call__(self, img1, img2):
        seed = torch.randint(0, 10000, (1,)).item()  # Generate a random seed

        torch.manual_seed(seed)  # Set seed before applying to first image
        img1 = self.transform(img1)

        torch.manual_seed(seed)  # Reset seed before applying to second image
        img2 = self.transform(img2)

        if self.debug:
            # Save the transformed images for debugging
            img1_save_path = os.path.join(self.save_dir, f"{seed}_img1_debug.png")
            img2_save_path = os.path.join(self.save_dir, f"{seed}_img2_debug.png")

            img1_pil = transforms.ToPILImage()(img1)  # Convert tensor back to PIL
            img2_pil = transforms.ToPILImage()(img2)

            img1_pil.save(img1_save_path)
            img2_pil.save(img2_save_path)


        return img1, img2

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

# Define a common transformation
paired_transform = PairedTransform(transform=train_transforms)
paired_transform_val = PairedTransform(transform=val_transforms)

dataset_root = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/"

file_names = sorted([f for f in os.listdir(os.path.join(dataset_root, "HR")) if f.startswith("sample") and f.endswith(".npy")])
train_files, val_files = train_test_split(file_names, test_size=0.1, random_state=42)

# Train MAE only on no_sub_train_files
batch_size=32
train_dataset = NPYSuperResolutionDataset(train_files, dataset_root, paired_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True)

# Validation set for MAE (later used in classification also)
val_dataset = NPYSuperResolutionDataset(val_files, dataset_root, paired_transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SuperResolutionViT(input_dim=input_dim, num_patches=num_patches).to(device)
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

# Track metrics
train_losses, val_losses = [], []
val_psnrs, val_ssims = [], []
train_psnrs, train_ssims = [], []

best_psnr = 0.0  # Track best PSNR
best_ssim = 0.0  # Track best SSIM

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for lr_images, hr_images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        optimizer.zero_grad()
        lr_images = image_to_patches(lr_images, 10)
        output = model(lr_images)

        loss = criterion(output.view(lr_images.shape[0], -1), hr_images.view(lr_images.shape[0], -1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        output = output.view(-1, 1, 150, 150)

        # Compute MSE on GPU
        mseall = F.mse_loss(output, hr_images, reduction='none')
        print(torch.mean(torch.sum(mseall, dim=[1, 2, 3])))
        print(loss)
        mse = torch.mean(mseall, dim=[1, 2, 3])
        batch_psnr = 10 * torch.log10(2.0 / mse)  # PSNR formula

        # Compute SSIM on GPU
        batch_ssim = ssim_torch(output, hr_images, data_range=2.0, size_average=False)  # (B,)

        total_psnr += batch_psnr.mean().item()
        total_ssim += batch_ssim.mean().item()

    train_loss = running_loss / len(train_loader)
    avg_psnr_train = total_psnr / len(train_loader)
    avg_ssim_train = total_ssim / len(train_loader)

    train_psnrs.append(avg_psnr_train)
    train_ssims.append(avg_ssim_train)
    train_losses.append(train_loss)

    # Validation Step
    model.eval()
    val_loss = 0.0
    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for lr_images, hr_images in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            lr_images = image_to_patches(lr_images, 10)
            output = model(lr_images)

            val_loss += criterion(output.view(lr_images.shape[0], -1), hr_images.view(lr_images.shape[0], -1)).item()
    
            output = output.view(-1, 1, 150, 150)
            # Compute MSE on GPU
            mse = torch.mean(F.mse_loss(output, hr_images, reduction='none'), dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10(2.0 / mse)  # PSNR formula

            # Compute SSIM on GPU
            batch_ssim = ssim_torch(output, hr_images, data_range=2.0, size_average=False)  # (B,)

            total_psnr += batch_psnr.mean().item()
            total_ssim += batch_ssim.mean().item()

    val_loss /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    val_losses.append(val_loss)
    val_psnrs.append(avg_psnr)
    val_ssims.append(avg_ssim)

    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train_PSNR: {avg_psnr_train:.2f} | Val_PSNR: {avg_psnr:.2f} | Train_SSIM: {avg_ssim_train:.4f} | Val_SSIM: {avg_ssim:.4f}")

    # Save Best Model based on PSNR
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "best_vit_PSNR_model.pth")
        print("Model Saved (Best PSNR)")

    # Save Best Model based on SSIM
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        torch.save(model.state_dict(), "best_vit_SSIM_model.pth")
        print("Model Saved (Best SSIM)")


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