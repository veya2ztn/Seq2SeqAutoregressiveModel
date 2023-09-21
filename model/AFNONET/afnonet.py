from functools import partial
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from .layers import Block
from ..PatchEmbedding import ConvPatchEmbed as PatchEmbed
from ..utils import transposeconv_engines
from ..base import DownAndUpModel,BaseModel
from ..model_arguments import AFNONetConfig



class AFNONet(DownAndUpModel):
    def __init__(self, config: AFNONetConfig):
        super().__init__(config)
        #### afnonet specific config
        ## ============================================
        ## create patch embedding model
        ## (B, P, W, H) -> (B, P, W//p, H//p)
        ## ============================================

        self.compress = self.build_PatchEmbedding(config)
        ## ============================================
        ## create position embedding
        ## (B, P, W//p, H//p)  -> (B, P, W//p, H//p)
        ## ============================================
        self.timepostion_information = nn.Identity()
        ## ============================================
        ## create main block
        ## (B, P, W//p, H//p)  -> (B, P, W//p, H//p)
        ## ============================================
        self.kernel = self.build_kernel(config)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.embed_dim)

        ## ============================================
        ## Upsample block: Make the output size match the input size
        ## (B, P, W//p, H//p)  -> (B, P, W, H)
        ## ============================================

        self.decompress = self.build_UpSampleBlock(config)

        ## ============================================
        ## Initialize
        ## ============================================
        self.apply(self.init_weights)

    @staticmethod
    def build_PatchEmbedding(config):
        history_length = config.history_length
        img_size       = config.img_size
        patch_size     = config.patch_size
        in_chans       = config.in_chans
        embed_dim      = config.embed_dim

        patch_size = [patch_size] * len(img_size) if isinstance(patch_size, int) else patch_size
        img_size   = (history_length, *img_size) if history_length > 1 else img_size
        patch_size = (1, *patch_size) if history_length > 1 else patch_size
        return PatchEmbed(img_size=img_size, patch_size=patch_size,
                          in_chans=in_chans, embed_dim=embed_dim)

    @staticmethod
    def build_kernel(config):
        drop_path_rate = config.get('drop_path_rate', 0.)
        mlp_ratio      = config.get('mlp_ratio', 4.)
        drop_rate      = config.get('drop_rate', 0.)
        double_skip    = config.get('double_skip', False)
        fno_bias       = config.get('fno_bias', False)
        fno_softshrink = config.get('fno_softshrink', False)
        uniform_drop   = config.get('uniform_drop', False)

        fno_blocks     = config.num_heads
        depth          = config.depth
        embed_dim      = config.embed_dim
        history_length = config.history_length

        img_size       = config.img_size
        img_size = (history_length, *img_size) if history_length > 1 else img_size
        
        dpr = [drop_path_rate]*depth if uniform_drop else np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        blocks = [Block(dim=embed_dim,
                                     mlp_ratio=mlp_ratio,
                                     drop=drop_rate,
                                     drop_path=dpr[i],
                                     norm_layer=norm_layer,
                                     double_skip=double_skip,
                                     fno_blocks=fno_blocks,
                                     fno_bias=fno_bias,
                                     fno_softshrink=fno_softshrink) for i in range(depth)]
        if   len(img_size)==2:blocks = [Rearrange('B P W H -> B W H P ')]    +blocks+[Rearrange('B W H P -> B P W H ')]
        elif len(img_size)==3:blocks = [Rearrange('B P Z W H -> B Z W H P ')]+blocks+[Rearrange('B Z W H P -> B P Z W H ')]
        elif len(img_size)==1:blocks = [Rearrange('B P W -> B W P ')]        +blocks+[Rearrange('B W P -> B P W ')]
        else:
            raise NotImplementedError
        return nn.Sequential(*blocks)

    @staticmethod
    def build_UpSampleBlock(config):
        history_length = config.history_length
        patch_size     = config.patch_size
        img_size       = config.img_size
        embed_dim      = config.embed_dim
        out_chans      = config.out_chans

        patch_size = [patch_size] * len(img_size) if isinstance(patch_size, int) else patch_size
        img_size   = (history_length, *img_size) if history_length > 1 else img_size
        patch_size = (1, *patch_size) if history_length > 1 else patch_size
        unique_up_sample_channel = config.get('unique_up_sample_channel')
        if unique_up_sample_channel is None:
            unique_up_sample_channel = config.out_chans
        conf_list = [{'kernel_size': [], 'stride':[], 'padding':[]},
                     {'kernel_size': [], 'stride':[], 'padding':[]},
                     {'kernel_size': [], 'stride':[], 'padding':[]}]
        conv_set = {8: [[2, 2, 0], [2, 2, 0], [2, 2, 0]],
                    4: [[2, 2, 0], [3, 1, 1], [2, 2, 0]],
                    2: [[3, 1, 1], [3, 1, 1], [2, 2, 0]],
                    1: [[3, 1, 1], [3, 1, 1], [3, 1, 1]],
                    3: [[3, 1, 1], [3, 1, 1], [3, 3, 0]],
                    }
        for patch in patch_size:
            for slot in range(len(conf_list)):
                conf_list[slot]['kernel_size'].append(conv_set[patch][slot][0])
                conf_list[slot]['stride'].append(conv_set[patch][slot][1])
                conf_list[slot]['padding'].append(conv_set[patch][slot][2])
        transposeconv_engine = transposeconv_engine = transposeconv_engines(
            len(img_size))
        last_Linear_layer = []
        if history_length > 1:
            if len(img_size) == 2:
                last_Linear_layer.append(
                    Rearrange('B P L H W -> B H W (P L) '))
            elif len(img_size) == 3:
                last_Linear_layer.append(
                    Rearrange('B P L Z H W -> B Z H W (P L) '))
            elif len(img_size) == 1:
                last_Linear_layer.append(Rearrange('B P L W -> B W (P L)'))
            else:
                raise NotImplementedError
        else:
            if len(img_size) == 2:
                last_Linear_layer.append(Rearrange('B P H W -> B H W P  '))
            elif len(img_size) == 3:
                last_Linear_layer.append(Rearrange('B P Z H W -> B Z H W P  '))
            elif len(img_size) == 1:
                last_Linear_layer.append(Rearrange('B P W -> B W P'))
            else:
                raise NotImplementedError
        last_Linear_layer.append(
            nn.Linear(out_chans*history_length, out_chans))
        if len(img_size) == 2:
            last_Linear_layer.append(Rearrange('B H W P-> B P H W'))
        elif len(img_size) == 3:
            last_Linear_layer.append(Rearrange('B Z H W P-> B P Z H W'))
        elif len(img_size) == 1:
            last_Linear_layer.append(Rearrange('B W P -> B P W'))
        else:
            raise NotImplementedError
        last_Linear_layer = nn.Sequential(*last_Linear_layer)

        return nn.Sequential(
            transposeconv_engine(
                embed_dim, unique_up_sample_channel*16, **conf_list[0]),
            nn.ReLU(),
            transposeconv_engine(unique_up_sample_channel*16,
                                 unique_up_sample_channel*4, **conf_list[1]),
            nn.ReLU(),
            transposeconv_engine(unique_up_sample_channel*4,
                                 out_chans,  **conf_list[2]),
            nn.ReLU(),
            last_Linear_layer
        )


