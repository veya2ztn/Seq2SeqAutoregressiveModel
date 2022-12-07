import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer



class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, use_time_embedding=False,  channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_z, image_height, image_width = self.triple(image_size)
        patch_z, patch_height, patch_width = self.triple(patch_size)

        assert image_z%patch_z == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, f'Image dimensions ({image_z},{image_height},{image_width}) must be divisible by the patch size ({patch_z},{patch_height},{patch_width}).'

        num_patches = (image_z//patch_z) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_z* patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (z p0) (h p1) (w p2) -> b (z h w) (p0 p1 p2 c)', p0 = patch_z, p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.use_time_embedding = use_time_embedding
        if self.use_time_embedding:
            self.time_embedding = nn.Linear(4, dim)
        self.dropout       = nn.Dropout(emb_dropout)
        self.transformer   = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def triple(self,t):
        return t if isinstance(t, tuple) else (t, t, t )

    def forward(self, img, time_stamp=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.use_time_embedding:
            time_embedding = self.use_time_embedding(time_stamp).unsqueeze(1)
            x = x + time_embedding #(B,T,L) + (B,1,L) -> (B,T,L)
        #x += self.pos_embedding[:, :(n + 1)] #remove time
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x)