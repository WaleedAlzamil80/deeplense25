'''import sys
import os
sys.path.append('/home/waleed/Documents/deeplense25/specific_test_06')
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.Dataset import NPYSuperResolutionDataset
from tqdm import tqdm
from utils.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches
from utils.helpful import print_trainable_parameters
from models.super_resolution import SuperResolutionViT


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

save_dir = "/home/waleed/Documents/deeplense25"
paired_transform = PairedTransform(transform=train_transforms)
paired_transform_val = PairedTransform(transform=val_transforms)

dataset_root = "/kaggle/input/deeplense/SpecificTest_06_B/Dataset"

file_names = sorted([f for f in os.listdir(os.path.join(dataset_root, "HR")) if f.startswith("sample") and f.endswith(".npy")])
train_files, val_files = train_test_split(file_names, test_size=0.1, random_state=42)

# Train MAE only on no_sub_train_files
batch_size=256
train_dataset = NPYSuperResolutionDataset(train_files, dataset_root, paired_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

# Validation set for MAE (later used in classification also)
val_dataset = NPYSuperResolutionDataset(val_files, dataset_root, paired_transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperResolutionViT(base="tiny", embed_dim = 192, input_dim=input_dim, num_patches=num_patches)
model = nn.DataParallel(model.to(device))

print_trainable_parameters(model)

base_model = "/home/waleed/Documents/deeplense25/specific_test_06/models/checkpoints/superresolution.pth"
state_dict = torch.load(base_model, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
'''
import sys
sys.path.append('/home/waleed/Documents/deeplense25/specific_test_06')
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_torch  # Fast SSIM on GPU
from sklearn.model_selection import train_test_split
from PIL import Image
from utils.Dataset import NPYSuperResolutionDataset
from utils.helpful import image_to_patches
from models.super_resolution import SuperResolutionViT
import torch.nn as nn
sns.set()

def evaluate_model(model_path, val_loader, device, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 10
    input_dim = patch_size**2
    num_patches = int(150/patch_size)**2
    
    model = SuperResolutionViT(input_dim=input_dim, num_patches=num_patches)
    model = nn.DataParallel(model.to(device))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    val_losses, val_psnrs, val_ssims = [], [], []
    with torch.no_grad():
        for lr_images, hr_images in tqdm(val_loader, desc='Evaluating Model'):
            plt.imsave("hr_image.png", hr_images[0].cpu().detach().view(150, 150).numpy(), cmap="gray")
            plt.imsave("lr_image.png", lr_images[0].cpu().detach().view(150, 150).numpy(), cmap="gray")
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            lr_images = image_to_patches(lr_images, patch_size)
            output = model(lr_images)
            plt.imsave("superResoluted.png", output.view(hr_images.shape)[0].cpu().detach().view(150, 150).numpy(), cmap="gray")

            # Compute MSE
            mse = F.mse_loss(output.view(hr_images.shape), hr_images, reduction='none')
            batch_psnr = 10 * torch.log10(2.0 / mse.mean(dim=[1, 2, 3]))
            batch_ssim = ssim_torch(output.view(hr_images.shape), hr_images, data_range=2.0, size_average=False)
            
            val_losses.append(mse.mean().item())
            val_psnrs.append(batch_psnr.mean().item())
            val_ssims.append(batch_ssim.mean().item())

    # Save results
    epochs = range(1, len(val_losses) + 1)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label='Val Loss', color='red')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Batches')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'SuperResolution_Val_Loss.png'))
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_psnrs, label='PSNR', color='green')
    plt.xlabel('Batch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Batches')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'SuperResolution_Val_PSNR.png'))
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_ssims, label='SSIM', color='blue')
    plt.xlabel('Batch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Batches')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'SuperResolution_Val_SSIM.png'))
    plt.show()
    
    print(f'Final Validation MSE: {sum(val_losses) / len(val_losses):.6f}')
    print(f'Final Validation PSNR: {sum(val_psnrs) / len(val_psnrs):.2f}')
    print(f'Final Validation SSIM: {sum(val_ssims) / len(val_ssims):.4f}')

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
    
if __name__ == "__main__":
    dataset_root = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/"
    file_names = sorted([f for f in os.listdir(os.path.join(dataset_root, "HR")) if f.startswith("sample") and f.endswith(".npy")])
    _, val_files = train_test_split(file_names, test_size=0.1, random_state=42)
    
    val_transforms = transforms.Compose([
        transforms.Resize(150, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    paired_transform_val = PairedTransform(transform=val_transforms)

    val_dataset = NPYSuperResolutionDataset(val_files, dataset_root, paired_transform_val)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=10, shuffle=False)
    
    save_dir = "/home/waleed/Documents/deeplense25"
    evaluate_model("/home/waleed/Documents/deeplense25/specific_test_06/models/checkpoints/best_vit_PSNR_model.pth", val_loader, "cuda", save_dir)