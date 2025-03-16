import torch
import os
import numpy as np
import matplotlib.pyplot as plt


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def image_to_patches(image, patch_size=16):
    """Convert a grayscale image into non-overlapping patches"""
    #transform = transforms.Compose([
    #    transforms.Resize((150, 150)),
    #    transforms.ToTensor(),          # Convert to Tensor
    #    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image
    #])

    #img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    img_tensor = image
    _, C, H, W = img_tensor.shape
    num_patches = (H // patch_size) * (W // patch_size)

    # Split into patches
    img_patches = img_tensor.unfold(2, patch_size, patch_size) # [1, 1, 224, 224] -> [1, 1, 14, 224, 16]
    img_patches = img_patches.unfold(3, patch_size, patch_size)  # [1, 1, 14, 224, 16] -> [1, 1, 14, 14, 16, 16]
    img_patches = img_patches.contiguous().view(-1, num_patches, patch_size * patch_size)  # Flatten patches [1, 1, 14, 14, 16, 16] -> [1, 196, 256]

    return img_patches


# Function to display images
def show_sample_images(dataset_path, class_names, num_samples=6):
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(10, 6))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        
        for j in range(num_samples):
            img = np.load(os.path.join(class_dir, files[j]), allow_pickle=True)
            if class_name == "axion":
                img = img[0]
            print(class_name, " ", img.shape)
            img = img.squeeze()  # Remove channel dim (1, H, W) -> (H, W)
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(class_name)

    plt.tight_layout()
    plt.show()

# Masking function
def random_masking(x, mask_ratio=0.75):
    """ Randomly masks a portion of patches from the input image tensor x """
    batch_size, num_patches, dim = x.shape
    num_masked = int(num_patches * mask_ratio)
    
    # Shuffle patch indices
    indices = torch.rand(batch_size, num_patches).argsort(dim=1)

    # Keep only a fraction of patches
    visible_indices = indices[:, num_masked:]
    masked_indices = indices[:, :num_masked]

    # Select the visible patches
    # print("gather: ", x.shape, " from dim = 1, Index: ", visible_indices.shape, visible_indices.unsqueeze(-1).shape, visible_indices.unsqueeze(-1).expand(-1, -1, dim).shape)
    # gather:  torch.Size([1, 196, 256])  from dim = 1, Index:  torch.Size([1, 49]) torch.Size([1, 49, 1]) torch.Size([1, 49, 256])
    # the specified dimention is the dimention that is repeated or extended other dimentions are fixed
    # so we gather from patches dimention 1

    visible_patches = torch.gather(x, dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, dim))

    return visible_patches, masked_indices, visible_indices

def visualize_patches(visible_patches, visible_indices, original_size=(150, 150), patch_size=16, title="Reconstructed Image from Visible Patches"):
    """
    Visualizes the visible patches by reconstructing them into a grid.
    
    Args:
        visible_patches (torch.Tensor): The output from `random_masking()`, shape (1, num_visible, patch_dim).
        visible_indices (torch.Tensor): The indices of the visible patches, shape (1, num_visible).
        original_size (tuple): The original image size (H, W).
        patch_size (int): The size of each patch (assumed square).
    """
    H, W = original_size
    num_patches_h = H // patch_size  # Number of patches along height
    num_patches_w = W // patch_size  # Number of patches along width

    # Initialize empty grid for reconstruction
    reconstructed_image = np.zeros((H, W))

    # Convert to numpy for easy manipulation
    visible_patches_np = visible_patches.squeeze(0).numpy()  # Remove batch dim
    visible_indices_np = visible_indices.squeeze(0).numpy()  # Remove batch dim

    for patch, idx in zip(visible_patches_np, visible_indices_np):
        # Get (row, col) location of patch in original grid
        row, col = divmod(idx, num_patches_w)
        
        # Reshape patch and place it in the correct location
        patch_img = patch.reshape(patch_size, patch_size)  # Reshape to 2D
        reconstructed_image[row * patch_size : (row + 1) * patch_size,
                            col * patch_size : (col + 1) * patch_size] = patch_img

    # Normalize for visualization
    reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())

    # Plot the reconstructed image
    plt.figure(figsize=(6, 6))
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()
