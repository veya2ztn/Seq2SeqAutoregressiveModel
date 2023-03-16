from model.afnonet import BaseModel
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

from networks.utils.utils import DropPath, PatchEmbed
from functools import partial
class PatchEmbedAlignModel(BaseModel):
    def __init__(self, low_level_img_size=None, low_level_patch_size=None, 
                 high_level_img_size=None, high_level_patch_size=None,
                 **kargs):
        super().__init__()
        norm_layer=None
        in_chans = kargs.get('in_chans',68)
        embed_dim=kargs.get('embed_dim',512)
        self.low_level_patch_embed  = PatchEmbed(patch_size=low_level_patch_size, in_c=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.high_level_patch_embed = PatchEmbed(
            patch_size=high_level_patch_size, in_c=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
    def forward(self, x1, x2):
        # x: [B, C, H, W]
        x1, T, H, W = self.low_level_patch_embed(x1)  # x:[B, H*W, C]
        x2, T, H, W = self.high_level_patch_embed(x2)  # x:[B, H*W, C]
        return x1, x2 


class PatchAlign_64_to_128(PatchEmbedAlignModel):
    def __init__(self, **kargs):
        super().__init__(low_level_img_size=(64, 128), low_level_patch_size=(1, 1), 
                  high_level_img_size = (128, 256), high_level_patch_size = (2, 2), **kargs)
        pretrain_weight = torch.load(
        "/nvme/zhangtianning/checkpoints/wpredict/lgnet_finetune/world_size8-swinvrnn64x128_lgnet_possloss_finetune_norandom/checkpoint_best.pth", map_location='cpu')['model']['lgnet']
        low_level_patch_embed_weight = {'proj.weight':pretrain_weight['module.net.patch_embed.proj.weight'],
                         'proj.bias': pretrain_weight['module.net.patch_embed.proj.bias']}
        self.low_level_patch_embed.load_state_dict(low_level_patch_embed_weight)
        high_level_patch_embed_weight = {'proj.weight':pretrain_weight['module.net.patch_embed.proj.weight'].repeat(1,1,2,2)/4,
                          'proj.bias': pretrain_weight['module.net.patch_embed.proj.bias']}
        self.high_level_patch_embed.load_state_dict(high_level_patch_embed_weight)
