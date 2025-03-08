import torch
import torch.nn as nn
import timm
from timm import create_model

from helpful import random_masking

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def modify_vit(vit_model):
    """
    Modifies a Vision Transformer (ViT) model.

    Args:
        vit_model: Pretrained ViT model from timm or torchvision.

    Returns:
        Modified ViT model.
    """
    # Get the patch embedding layer
    patch_embed = vit_model.patch_embed

    # Create a new layer with input channel changed from 3 → 1
    # new_patch_embed = torch.nn.Conv2d(
    #     in_channels=1,  # Change input channels to 1 (grayscale)
    #     out_channels=patch_embed.proj.out_channels,  # Keep the same output channels
    #     kernel_size=patch_embed.proj.kernel_size,  # Keep the kernel size
    #     stride=patch_embed.proj.stride,  # Keep the stride
    #     padding=patch_embed.proj.padding,  # Keep the padding
    #     bias=patch_embed.proj.bias is not None  # Keep bias settings
    # )
    new_patch_embed = nn.Linear(256, 768)

    # Copy weights by averaging across RGB channels
    with torch.no_grad():
        new_patch_embed.weight[:] = patch_embed.proj.weight.mean(dim=1, keepdim=True)

    # Replace the old patch embedding layer
    vit_model.patch_embed.proj = new_patch_embed

    return vit_model

class Classifiervit(nn.Module):
    def __init__(self, num_classes=3, p=0.5):
        super(Classifiervit, self).__init__()

        self.model = modify_vit(create_model("vit_tiny_patch16_224", pretrained=True))

        self.fc1 = nn.Linear(1000, 128)
        # self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.model(x)
        x = self.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn(self.fc2(x)))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

# Define the Encoder (ViT model)
class MAEViT(nn.Module):
    def __init__(self, encoder, embed_dim=768, num_patches=196, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Apply masking
        visible_patches, masked_indices, visible_indices = random_masking(x, self.mask_ratio)
        
        # Encode only visible patches
        encoded = self.encoder.encoder(visible_patches)
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return reconstructed, masked_indices
