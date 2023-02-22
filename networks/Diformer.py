import torch.nn as nn
import torch
from networks.utils.utils import DropPath, PatchEmbed
import copy
from networks.utils.utils import Mlp
from functools import partial
from networks.utils.Attention import SD_attn
import torch.utils.checkpoint as checkpoint
from networks.utils.utils import DropPath, window_partition, window_reverse, ScaleOffset, attn_norm





def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)


class Block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.ReLU, norm_layer=nn.LayerNorm, depth=0, swin=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if depth % 2 == 0:
            dilated_size = [1, 1, 1]
        else:
            dilated_size = [2**((depth+1)//2), 2**((depth+1)//2), 2**((depth+1)//2)]
        dilated_size = dilated_size[-len(window_size):]

        if depth % 2 == 1:
            self.shift_size = [(dilated_size[i] + 1) * window_size[i]//2 for i in range(len(window_size))]
        else:
            self.shift_size = [0 for _ in window_size]

        
        self.norm = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
        self.attn = SD_attn(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, shift_size=self.shift_size, dilated_size=dilated_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    

    def forward(self, x):
        

        shortcut = x
        x = self.norm(x)
        # partition windows

        
        # W-MSA/SW-MSA
        x = self.attn(x)  # [B, T, H, W, C]

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))


        return x

class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, only_swin=False) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        # for i in range(len(window_size)):
        #     if window_size[-i] == img_size[-i]:
        #         self.shift_size[-i] = 0


        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                depth=i,
                swin=only_swin
            )
            for i in range(depth)
        ])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
    
        return x




class SWDfromer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False, only_swin=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = Layer(dim=embed_dim,
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                only_swin=only_swin
                                )
            self.layers.append(layers)

        self.final = nn.Linear(embed_dim, out_chans, bias=False)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, 32*64, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        x, T, H, W = self.patch_embed(x)  # x:[B, H*W, C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)

        for layer in self.layers:
            x= layer(x)

        x = self.final(x)
        if len(self.window_size) == 3:
            res = x.permute(0, 4, 1, 2, 3)
        elif len(self.window_size) == 2:
            res = x.permute(0, 3, 1, 2)

        # x = self.norm(x)  # [B, L, C]
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return res


class Diformer(nn.Module):
    def __init__(self, patch_size=(1,1,1), in_chans=20, out_chans=20, embed_dim=768, window_size=[4,8], depths=[2, 2, 6, 2], \
                num_heads=[3, 6, 12, 24], Weather_T=16, only_swin=False) -> None:
        super().__init__()
        self.net = SWDfromer(patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, \
                                        embed_dim=embed_dim, depths=depths, num_heads=num_heads, \
                                        window_size=window_size, only_swin=only_swin)
    


    def forward(self, data):
        out = self.net(data)

        return out
