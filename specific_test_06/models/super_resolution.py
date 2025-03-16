import math
import torch
from torch import nn
from mae import EncoderViT, MAEDecoder

# Define the Encoder (ViT model)
class SuperResolutionViT(nn.Module):
    def __init__(self, base="tiny", input_dim=256, embed_dim=192, num_patches=196, p=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedInput = nn.Linear(input_dim, self.embed_dim)
        self.encoder = EncoderViT(base=base, p=p)
        self.decoder = MAEDecoder(output_patch=input_dim, embed_dim=embed_dim)

        self.num_patches = num_patches

        # Compute and store full position encoding ONCE
        self.register_buffer("full_position_encoding", self.sinusoidal_position_encoding(num_patches, self.embed_dim).unsqueeze(0))
        # self.full_position_encoding = nn.Parameter(torch.randn(1, num_patches, self.embed_dim))  # Learnable

    def forward(self, x):
        batch_size = x.shape[0]
        full_pos_encoding = self.full_position_encoding.expand(batch_size, -1, -1)

        x = self.embedInput(x) + full_pos_encoding # (bs, visible_patches, embed_dim)

        # Encode only visible patches
        encoded = self.encoder(x)

        # Decode
        full_pos_encoding = self.full_position_encoding.expand(batch_size, -1, -1)  # Shape: [batch_size, 196, embed_dim]
        full_tokens = encoded + full_pos_encoding

        reconstructed = self.decoder(full_tokens)
        return reconstructed

    def sinusoidal_position_encoding(self, num_patches, embed_dim):
        position = torch.arange(num_patches).unsqueeze(1)  # Shape: (num_patches, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)

        return pe  # Shape: (num_patches, embed_dim)
