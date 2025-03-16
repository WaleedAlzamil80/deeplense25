import math
import torch
import torch.nn as nn
from timm import create_model

class EncoderViT(nn.Module):
    def __init__(self, base="tiny", p=0.25):
        super(EncoderViT, self).__init__()

        modelss = create_model(f"vit_{base}_patch16_224", pretrained=True)
        modelss.patch_embed = nn.Identity()
        modelss.head = nn.Identity() # now output shape is embed_dim (tiny: 192, base: 768)
        self.set_dropout(modelss, p)

        self.encoder_blocks = modelss.blocks
        self.norm = modelss.norm

    def set_dropout(self, model, p):
        """Recursively set dropout probability in a model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        return x

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=192, output_patch=256, num_layers=2, num_heads=6):
        super().__init__()

        # Transformer decoder layers
        self.decoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(embed_dim, output_patch)  # Output patch embeddings

    def forward(self, x):
        x = self.decoder_blocks(x)  # Apply transformer decoder layers
        return self.final_layer(x)  # Shape: [batch_size, num_patches, embed_dim]


# Define the Encoder (ViT model)
class MAEViT(nn.Module):
    def __init__(self, base="tiny", input_dim=256, embed_dim=192, num_patches=196, mask_ratio=0.75, p=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedInput = nn.Linear(input_dim, self.embed_dim)
        self.encoder = EncoderViT(base=base, p=p)
        self.decoder = MAEDecoder(output_patch=input_dim, embed_dim=embed_dim)

        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.num_masked = int(self.num_patches * self.mask_ratio)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable mask token

        # Compute and store full position encoding ONCE
        self.register_buffer("full_position_encoding", self.sinusoidal_position_encoding(num_patches, self.embed_dim).unsqueeze(0))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Apply masking
        visible_patches, masked_indices, visible_indices = self.random_masking(x)

        visible_pos_encoding = torch.gather(
            self.full_position_encoding.expand(x.shape[0], -1, -1), dim=1, 
            index=visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        x = self.embedInput(visible_patches) + visible_pos_encoding # (bs, visible_patches, embed_dim)

        # Encode only visible patches
        encoded = self.encoder(x)

        # Decode
        ## masking
        ### Create a full set of tokens filled with MASK tokens
        full_tokens = self.mask_token.repeat(batch_size, self.num_patches, 1)
        ### Insert the encoded visible patches into the correct positions
        full_tokens.scatter_(1, visible_indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]), encoded)
        ### Apply position encoding to all patches (both visible and masked)
        full_pos_encoding = self.full_position_encoding.expand(batch_size, -1, -1)  # Shape: [batch_size, 196, embed_dim]
        full_tokens = full_tokens + full_pos_encoding

        reconstructed = self.decoder(full_tokens)
        # plt.imsave("reconstructed_image.png", reconstructed[0].cpu().detach().view(150, 150).numpy(), cmap="gray")
        return reconstructed, masked_indices

    def sinusoidal_position_encoding(self, num_patches, embed_dim):
        position = torch.arange(num_patches).unsqueeze(1)  # Shape: (num_patches, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)

        return pe  # Shape: (num_patches, embed_dim)

    def random_masking(self, x):
        batch_size, num_patches, dim = x.shape

        # Shuffle patch indices
        indices = torch.rand(batch_size, num_patches).argsort(dim=1).to(x.device)
    
        # Keep only a fraction of patches
        visible_indices = indices[:, self.num_masked:]
        masked_indices = indices[:, :self.num_masked]
    
        # Select the visible patches
        visible_patches = torch.gather(x, dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, dim))
    
        return visible_patches, masked_indices, visible_indices