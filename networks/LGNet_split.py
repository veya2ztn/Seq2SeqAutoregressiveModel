import torch.nn as nn
import torch
from networks.utils.utils import DropPath, PatchEmbed
import copy
from networks.utils.utils import Mlp
from functools import partial
from networks.utils.Attention import SD_attn
from networks.utils.Blocks import Convnet_block, Windowattn_block
from einops import rearrange
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from networks.SwinUnet_3d import SwinTransformer



class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, layer_type="convnet_block",
                use_checkpoint=False) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        # for i in range(len(window_size)):
        #     if window_size[-i] == img_size[-i]:
        #         self.shift_size[-i] = 0

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if layer_type == "convnet_block":
                block = Convnet_block(
                        dim=dim,
                        kernel_size=window_size,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale_init_value = 0,
                        norm_layer=norm_layer,
                    )
            elif layer_type == "window_block":
                block = Windowattn_block(
                        dim=dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                    )
            elif layer_type == "swin_block":
                block = Windowattn_block(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    shift_size=[0,0] if i%2==0 else [i//2 for i in window_size]
                )
            if use_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=True)
            self.blocks.append(block)

        # if layer_type == "convnet_block":
        #     self.blocks = nn.ModuleList([
        #         Convnet_block(
        #             dim=dim,
        #             kernel_size=window_size,
        #             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #             layer_scale_init_value = 0,
        #             norm_layer=norm_layer,
        #         )
        #         for i in range(depth)
        #     ])
        # elif layer_type == "window_block":
        #     self.blocks = nn.ModuleList([
        #     Windowattn_block(
        #         dim=dim,
        #         window_size=window_size,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_layer=norm_layer,
        #     )
        #     for i in range(depth)
        # ])
        # elif layer_type == "swin_block":
        #     self.blocks = nn.ModuleList([
        #     Windowattn_block(
        #         dim=dim,
        #         window_size=window_size,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_layer=norm_layer,
        #         shift_size=[0,0] if i%2==0 else [i//2 for i in window_size]
        #     )
        #     for i in range(depth)
        # ])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
    
        return x

class Enc_net(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=[4], out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        # split image into non-overlapping patches
        self.transformer_dict = nn.ModuleList()
        for in_chan in in_chans:
            self.transformer_dict.append(SwinTransformer(img_size, patch_size=(2, 2), in_chans=in_chan,
                                                        embed_dim=96, depths=(2, 2, 2), num_heads=(3, 6, 12),
                                                        window_size=(6, 12), mlp_ratio=4., qkv_bias=True,
                                                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                                        norm_layer=nn.LayerNorm, patch_norm=True,
                                                        use_checkpoint=use_checkpoint))
        self.fc = nn.Linear(in_features=96*4*7, out_features=embed_dim, bias=True)

    def forward(self, x):
        split_x = torch.split(x, self.in_chans, dim=1)
        split_out_x = []
        for i in range(len(self.in_chans)):
            split_out_x.append(self.transformer_dict[i](split_x[i])[-1])
        out_x = torch.cat(split_out_x, dim=1).permute(0, 2, 3, 1)
        out = self.fc(out_x).permute(0, 3, 1, 2)
        return out
            




class LG_net(nn.Module):

    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
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
                                window_size=[img_size[-2]//patch_size[-2], img_size[-1]//patch_size[-1]] if i_layer==0 else window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                layer_type="window_block" if i_layer==0 else "swin_block",
                                use_checkpoint=use_checkpoint
                                )
            self.layers.append(layers)

        # self.final = nn.Linear(embed_dim, out_chans, bias=False)
        self.final = nn.Linear(embed_dim, out_chans*patch_size[-1]*patch_size[-2], bias=False)
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]//patch_size[-2]*img_size[1]//patch_size[-1], embed_dim))
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
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        T = 1
        # x, T, H, W = self.patch_embed(x)  # x:[B, H*W, C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)

        for layer in self.layers:
            x= layer(x)

        x = self.final(x)

        res = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[-2],
            p2=self.patch_size[-1],
            h=self.img_size[0] // self.patch_size[-2],
            w=self.img_size[1] // self.patch_size[-1],
        )

        # if len(self.window_size) == 3:
        #     res = x.permute(0, 4, 1, 2, 3)
        # elif len(self.window_size) == 2:
        #     res = x.permute(0, 3, 1, 2)

        # x = self.norm(x)  # [B, L, C]
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return res



class LGNet_split(nn.Module):
    def __init__(self, img_size=[32, 64], patch_size=(1,1,1), in_chans=20, out_chans=20, embed_dim=768, window_size=[4,8], depths=[2, 2, 6, 2], \
                num_heads=[3, 6, 12, 24], Weather_T=16, drop_rate=0., attn_drop_rate=0., drop_path=0., use_checkpoint=False) -> None:
        super().__init__()
        self.enc = Enc_net(img_size=img_size, patch_size=(2, 2), in_chans=[6, 37, 37, 37, 37, 37, 37],
                 embed_dim=embed_dim, depths=(2, 2, 2), num_heads=(3, 6, 12),
                 window_size=window_size, mlp_ratio=4., use_checkpoint=use_checkpoint)
        self.net = LG_net(img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, \
                                        embed_dim=embed_dim, depths=depths, num_heads=num_heads, \
                                        window_size=window_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                                        drop_path_rate=drop_path, use_checkpoint=use_checkpoint)
    


    def forward(self, data, **kwargs):
        data = self.enc(data)
        out = self.net(data)

        return out