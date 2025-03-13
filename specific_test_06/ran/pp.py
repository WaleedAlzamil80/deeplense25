import os
from PIL import Image
import sys
sys.path.append('/home/waleed/Documents/deeplense25')

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lpips  # Perceptual similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import seaborn as sns
from specific_test_06.ran.helpful import image_to_patches, show_sample_images, random_masking, visualize_patches
sns.set()

# Convert PyTorch tensor to PIL Image
def tensor_to_pil(img_tensor):
    img_array = img_tensor.squeeze().cpu().numpy()  # Remove batch dim, convert to NumPy
    return Image.fromarray((img_array * 255).astype("uint8"))  # Scale back to 0-255

# Convert PIL Image back to PyTorch tensor
def pil_to_tensor(img_pil):
    return transforms.ToTensor()(img_pil).unsqueeze(0)  # Add batch dimension

# Show samples
class_namesB = ["LR", "HR"]
dataset_pathB = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset"
show_sample_images(dataset_pathB, class_namesB)
class_namesA = ["cdm", "no_sub", "axion"]
dataset_pathA = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset"
show_sample_images(dataset_pathA, class_namesA)

# Load an example image (grayscale from .npy)
image_path = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_A/Dataset/no_sub/no_sub_sim_100008917700138988486624773794898508448.npy"
image_path = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/LR/sample100.npy"
image_path2 = "/home/waleed/Downloads/GSoC25_ML4SC/SpecificTest_06_B/Dataset/HR/sample100.npy"

image_array = np.load(image_path)  # Load .npy file
image_tensor = torch.tensor(image_array, dtype=torch.float32)  # Convert to tensor
image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  # Normalize


img = image_tensor.unsqueeze(0)

# Upscale to (150,150) using different interpolation methods
nearest = F.interpolate(img, size=(150, 150), mode="nearest")
bilinear = F.interpolate(img, size=(150, 150), mode="bilinear", align_corners=False)
bicubic = F.interpolate(img, size=(150, 150), mode="bicubic", align_corners=False)
area = F.interpolate(img, size=(150, 150), mode="area")
nearest_exact = F.interpolate(img, size=(150, 150), mode="nearest-exact")
# Convert to PIL Image
img_pil = tensor_to_pil(img)
# Apply Lanczos Interpolation (Resize 64x64 → 150x150)
lanczos_img_pil = img_pil.resize((150, 150), Image.LANCZOS)
# Convert back to PyTorch tensor
lanczos = pil_to_tensor(lanczos_img_pil)



# Convert tensors to numpy for visualization
def to_numpy(tensor):
    return tensor.squeeze().detach().cpu().numpy()


#axs[3].imshow(to_numpy(lanczos), cmap="gray"); axs[3].set_title("Lanczos") linear | trilinear | area | nearest-exact

# Plot results
fig, axs = plt.subplots(3, 3, figsize=(20, 20))  # Create a 2x3 grid

# Flatten the 2D array of axes for easier iteration
axs = axs.flatten()

# Plot each interpolation method in the grid
axs[1].imshow(to_numpy(img), cmap="gray"); axs[1].set_title("Original")
axs[0+3].imshow(to_numpy(lanczos), cmap="gray"); axs[0+3].set_title("Lanczos")
axs[1+3].imshow(to_numpy(bilinear), cmap="gray"); axs[1+3].set_title("Bilinear")
axs[2+3].imshow(to_numpy(bicubic), cmap="gray"); axs[2+3].set_title("Bicubic")
axs[3+3].imshow(to_numpy(area), cmap="gray"); axs[3+3].set_title("Area")
axs[4+3].imshow(to_numpy(nearest_exact), cmap="gray"); axs[4+3].set_title("Nearest-Exact")
axs[5+3].imshow(to_numpy(nearest), cmap="gray"); axs[5+3].set_title("Nearest Neighbor")

# Remove any empty subplot borders (optional)
for ax in axs:
    ax.axis("off")

plt.tight_layout()  # Adjust layout for better spacing
plt.show()





# Load LPIPS model (used for perceptual similarity)
lpips_model = lpips.LPIPS(net='alex')  # Uses AlexNet features

# Convert to tensor format for LPIPS
def to_lpips_tensor(img):
    return torch.tensor(img).unsqueeze(0).expand(1, 3, *img.shape[-2:])  # Convert to 3-channel

# Function to compute all evaluation metrics
def evaluate_upscaling(original, upscaled):
    """
    Evaluates upscaled image quality using MSE, PSNR, SSIM, and LPIPS.
    
    Args:
        original (numpy array): Original image (grayscale)
        upscaled (numpy array): Upscaled image (grayscale)
        
    Returns:
        dict: Dictionary with computed metric values
    """

    # Convert to float32 for accuracy
    original = original.astype(np.float32) / 255.0
    upscaled = upscaled.astype(np.float32) / 255.0

    # Compute MSE
    mse = np.mean((original - upscaled) ** 2)

    # Compute PSNR
    psnr_value = psnr(original, upscaled, data_range=1.0)

    # Compute SSIM
    ssim_value = ssim(original, upscaled, data_range=1.0)

    # Compute LPIPS (deep learning perceptual similarity)
    lpips_value = lpips_model(to_lpips_tensor(original), to_lpips_tensor(upscaled)).item()

    return {
        "MSE": mse,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "LPIPS": lpips_value
    }

# Function to display metric results
def display_results(results):
    print("\n🔹 **Evaluation Metrics for Upscaling Methods** 🔹")
    for method, scores in results.items():
        print(f"\n📌 **{method}**")
        for metric, value in scores.items():
            print(f"   {metric}: {value:.4f}")

image_array = np.load(image_path2)  # Load .npy file
image_tensor = torch.tensor(image_array, dtype=torch.float32)  # Convert to tensor
image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  # Normalize
img = image_tensor.unsqueeze(0).unsqueeze(0)

# Convert images to numpy arrays
original_np = to_numpy(img)  # Original Image
methods = {
    "Lanczos": to_numpy(lanczos),
    "Bilinear": to_numpy(bilinear),
    "Bicubic": to_numpy(bicubic),
    "Area": to_numpy(area),
    "Nearest-Exact": to_numpy(nearest_exact),
    "Nearest": to_numpy(nearest)
}

# Compute metrics for each method
results = {method: evaluate_upscaling(original_np, upscaled_np) for method, upscaled_np in methods.items()}

# Display results
display_results(results)

# Plot results in bar charts
metrics = ["MSE", "PSNR", "SSIM", "LPIPS"]
fig, axs = plt.subplots(1, 4, figsize=(18, 5))

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in methods.keys()]
    axs[i].bar(methods.keys(), values)
    axs[i].set_title(metric)
    axs[i].set_xticklabels(methods.keys(), rotation=45)

plt.tight_layout()
plt.show()
