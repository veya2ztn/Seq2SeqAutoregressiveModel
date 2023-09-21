from .utils import conv_engines
import torch
import torch.nn as nn
import numpy as np

class ConvPatchEmbed(nn.Module):
    """
    A CNN patch embedding layer
    """

    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768, conv_simple=True):
        super().__init__()

        if img_size is None:raise KeyError('img is None')
        patch_size = [patch_size] * len(img_size) if isinstance(patch_size, int) else patch_size

        num_patches = 1
        out_size = []
        for i_size, p_size in zip(img_size, patch_size):
            if not i_size % p_size:
                num_patches *= i_size // p_size
                out_size.append(i_size // p_size)
            else:
                raise NotImplementedError(
                    f"the patch size ({patch_size}) cannot divide the img size {img_size}")
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.num_patches = num_patches
        self.out_size = tuple(out_size)
        #conv_engine = [nn.Conv1d,nn.Conv2d,nn.Conv3d]
        conv_engine = conv_engines(len(img_size), conv_simple=conv_simple)
        self.proj   = conv_engine(in_chans, embed_dim,
                                kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, = x.shape[:2]
        #print(x.shape)
        inp_size = tuple(x.shape[2:])
        assert tuple(inp_size) == self.img_size, f"Input image size ({inp_size}) doesn't match model set size ({self.img_size})."
        x = self.proj(x)
        return x
