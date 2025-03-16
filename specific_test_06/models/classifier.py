import math
import torch
from torch import nn
from mae import EncoderViT, MAEDecoder

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))  # Learnable query vector
        self.attn_weights = nn.Linear(embed_dim, 1)  # Linear layer to compute scores

    def forward(self, x):
        """
        x: (batch_size, num_patches, embed_dim)
        Returns: (batch_size, embed_dim) - aggregated representation
        """
        scores = self.attn_weights(x).squeeze(-1)  # (batch_size, num_patches)
        attn_weights = torch.softmax(scores, dim=1)  # Normalize
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return pooled

# Define the Encoder (ViT model)
class ClassifierViT(nn.Module):
    def __init__(self, base="tiny", input_dim=256, embed_dim=192, num_patches=196, p=0.25, num_classes=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedInput = nn.Linear(input_dim, self.embed_dim)
        self.encoder = EncoderViT(base=base, p=p)
        self.num_patches = num_patches
        # Load saved components
        ## checkpoint = torch.load("/kaggle/input/specific_task_06/pytorch/default/2/encoder_embedInput.pth", weights_only=True)
        ## self.embedInput.load_state_dict(checkpoint["embedInput"])
        ## self.encoder.load_state_dict(checkpoint["encoder"])

        self.attnetion_pool = AttentionPooling(embed_dim)
        self.num_patches = num_patches
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, num_classes)
        )

        self.register_buffer("full_position_encoding", self.sinusoidal_position_encoding(num_patches, self.embed_dim).unsqueeze(0))

    def forward(self, x):
        batch_size = x.shape[0]
        
        full_pos_encoding = self.full_position_encoding.expand(batch_size, -1, -1)
        x = self.embedInput(x) + full_pos_encoding # (bs, visible_patches, embed_dim)
        
        x = self.encoder(x)
        x = self.attnetion_pool(x)
        x = self.classifier(x)
        return x

    def sinusoidal_position_encoding(self, num_patches, embed_dim):
        position = torch.arange(num_patches).unsqueeze(1)  # Shape: (num_patches, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)

        return pe  # Shape: (num_patches, embed_dim)
