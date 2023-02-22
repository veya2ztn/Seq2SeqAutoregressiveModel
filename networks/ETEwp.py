""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
Code/weights from https://github.com/microsoft/Swin-Transformer
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import math
from networks.utils.utils import window_partition, window_reverse, DropPath, PatchEmbed
from networks.utils.positional_encodings import rope2, rope3
import copy
from networks.utils.mlp import Mlp
from functools import partial
from networks.utils.Blocks import Convnet_block, Windowattn_block



def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class time_embed(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(time_embed, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor, pos):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """

        # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #     return self.cached_penc

        self.cached_penc = None
        
        batch_size, N, orig_ch = tensor.shape

        pos_x = pos[:,0]
        pos_y = pos[:,1]
        pos_z = pos[:,2]

        sin_inp_x = torch.einsum("i,k->ik", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,k->ik", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,k->ik", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((batch_size, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, : self.channels] = emb_x
        emb[:, self.channels : 2 * self.channels] = emb_y
        emb[:, 2 * self.channels :] = emb_z


        res = tensor + emb.unsqueeze(1)
        return res




class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, layer_type="convnet_block") -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        # for i in range(len(window_size)):
        #     if window_size[-i] == img_size[-i]:
        #         self.shift_size[-i] = 0
        if layer_type == "convnet_block":
            self.blocks = nn.ModuleList([
                Convnet_block(
                    dim=dim,
                    kernel_size=window_size,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale_init_value = 0,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])
        elif layer_type == "window_block":
            self.blocks = nn.ModuleList([
            Windowattn_block(
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
            for i in range(depth)
        ])
        elif layer_type == "swin_block":
            self.blocks = nn.ModuleList([
            Windowattn_block(
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
            for i in range(depth)
        ])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
    
        return x







class LG_Encoder(nn.Module):

    def __init__(self, patch_size=(1, 1, 1), img_size=[32,64], in_chans=3,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,
                 Encoder_use_time_info=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.Encoder_use_time_info = Encoder_use_time_info

        if Encoder_use_time_info:
            self.time_embed = time_embed(embed_dim)
        else:
            self.time_embed = None

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
            layers = Layer(dim = embed_dim,
                            depth = depths[i_layer],
                            num_heads = num_heads[i_layer],
                            window_size = img_size if i_layer==0 else window_size,
                            mlp_ratio = self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            norm_layer=norm_layer,
                            layer_type="window_block" if i_layer==0 else "swin_block"
                            )
            self.layers.append(layers)

        # self.final = nn.Linear(embed_dim, out_chans, bias=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[-2]*img_size[-1], embed_dim))
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

    def forward(self, x, time_info):
        # x: [B, C, H, W]
        B = x.shape[0]
        x, T, H, W = self.patch_embed(x)  # x:[B, H*W, C]
        if self.time_embed != None:
            x = self.time_embed(x, time_info)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)

        res = []

        for layer in self.layers:
            x= layer(x)
            res.append(x)

        # if len(self.window_size) == 3:
        #     res = x.permute(0, 4, 1, 2, 3)
        # elif len(self.window_size) == 2:
        #     res = x.permute(0, 3, 1, 2)

        return res


   

class Cross_attn(nn.Module):
    def __init__(self, dim, x_resolution, y_resolution, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution


        self.position_enc1 = rope2(self.x_resolution, self.head_dim)
        
        self.position_enc2 = rope2(self.y_resolution, self.head_dim)


        self.l_q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.l_kv = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.l_proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):

        B, H_x, W_x, C = x.shape
        B, H_y, W_y, C = y.shape

        q = self.l_q(x).reshape(B, H_x * W_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.l_kv(y).reshape(B, H_y * W_y, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H_x, W_x, self.dim)
        x = self.l_proj(x)
        x = self.proj_drop(x)
        return x



class Cross_block(nn.Module):
    def __init__(self, dim, window_size, img_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm

        self.norm = norm_layer(dim)

        self.cross_attn = Cross_attn(
            dim, img_size, img_size, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)



    def forward(self, x, y):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = shortcut + self.drop_path(self.cross_attn(self.norm(x), y))
        else:
            x = self.norm(shortcut + self.drop_path(self.cross_attn(x, y)))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))


        return x

class Decoder_Layer(nn.Module):
    def __init__(self, dim, depth, window_size, img_size,
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, cross_first = True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.cross_first=cross_first

        self.blocks = nn.ModuleList()
        self.cross_block = Cross_block(dim=dim,
                                window_size=window_size,
                                img_size=img_size,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop,
                                attn_drop=attn_drop,
                                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)



        for i in range(depth):
            self.blocks.append(
                Windowattn_block(
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
            )


    def forward(self, x, memory):
        if self.cross_first:
            x = self.cross_block(x, memory)
            for blk in self.blocks:
                x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.cross_block(x, memory)
        return x

class SwinDecoder(nn.Module):

    def __init__(self, img_size, patch_size=[1,1], out_chans=20,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, cross_first=True, 
                 query_nums=1, use_time_info=False, use_geo_info=False,
                 geo_inchans=0, patch_norm=False):
        super().__init__()

        self.num_layers = len(depths)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.query_nums = query_nums
        self.use_time_info = use_time_info
        self.use_geo_info = use_geo_info

        if use_time_info:
            self.time_embed = time_embed(embed_dim)
        else:
            self.time_embed = None


        total_query_dim = img_size[-2] * img_size[-1] * embed_dim


        if use_geo_info:
            self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=geo_inchans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = None

        if query_nums == 1:
            self.random_query = nn.init.trunc_normal_(nn.Parameter(torch.zeros(*img_size[-2:], embed_dim)), 0., 0.02)
        else:
            self.random_query = nn.Embedding(query_nums, embedding_dim=total_query_dim)
        # split image into non-overlapping patches

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layer = Decoder_Layer(dim = embed_dim,
                            depth = depths[i_layer],
                            num_heads = num_heads[i_layer],
                            window_size = window_size,
                            img_size = img_size,
                            mlp_ratio = self.mlp_ratio,
                            qkv_bias = qkv_bias,
                            drop = drop_rate,
                            attn_drop = attn_drop_rate,
                            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            norm_layer = norm_layer,
                            cross_first = cross_first
                            )
            self.layers.append(layer)

        self.final = nn.Linear(embed_dim, out_chans)



        

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, memory, index=None, time_info=None, geo_info=None):
        B = memory.shape[0]
        if self.query_nums == 1:
            x = self.random_query.repeat(B, 1, 1, 1)
        else:
            x = self.random_query(index).reshape(B, self.img_size[-2], self.img_size[-1], self.embed_dim)

        B, H, W, _ = x.shape
        if self.use_geo_info:
            geo_embed, T, H, W = self.patch_embed(geo_info)  # x:[B, H*W, C]
            x = x.reshape(B, -1, self.embed_dim) + geo_embed
        if self.use_time_info:
            x = self.time_embed(x.reshape(B, -1, self.embed_dim), time_info)

        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)

        for layer in self.layers:
            x= layer(x, memory)

        x = self.final(x)
        if len(self.window_size) == 3:
            res = x.permute(0, 4, 1, 2, 3)
        elif len(self.window_size) == 2:
            res = x.permute(0, 3, 1, 2)
        return res
        



class ETEwp(nn.Module):
    def __init__(self, 
                img_size = [32,64], 
                patch_size = (1,1,1), 
                in_chans = 20, 
                out_chans = 20, 
                embed_dim = 96, 
                window_size = [4,8], 
                encoder_depths = [4, 4, 4],
                encoder_heads = [6, 6, 6], 
                decoder_depths = [4, 4],
                decoder_heads = [6, 6],
                Weather_T = 1,
                drop_rate = 0.,
                attn_drop_rate = 0., 
                drop_path_rate = 0.,
                cross_first = True,
                Encoder_use_time_info=False,
                query_nums=1,
                Decoder_use_time_info=False,
                Decoder_use_geo_info=False,
                geo_inchans=2) -> None:
        super().__init__()
        self.encoder = LG_Encoder(patch_size = patch_size,
                                img_size = img_size, 
                                in_chans = in_chans,
                                embed_dim = embed_dim, 
                                depths = encoder_depths, 
                                num_heads = encoder_heads,
                                window_size = window_size, 
                                drop_rate = drop_rate, 
                                attn_drop_rate = attn_drop_rate, 
                                drop_path_rate = drop_path_rate, 
                                patch_norm = False,
                                Encoder_use_time_info=Encoder_use_time_info)
        self.decoder = SwinDecoder(img_size = img_size, 
                                    patch_size = patch_size,
                                    out_chans = out_chans,
                                    embed_dim = embed_dim, 
                                    depths = decoder_depths, 
                                    num_heads = decoder_heads,
                                    window_size = window_size, 
                                    drop_rate = drop_rate, 
                                    attn_drop_rate = attn_drop_rate, 
                                    drop_path_rate = drop_path_rate,
                                    cross_first = cross_first,
                                    query_nums=query_nums,
                                    use_time_info=Decoder_use_time_info, 
                                    use_geo_info=Decoder_use_geo_info,
                                    geo_inchans=geo_inchans
                                    )


    def forward(self, data, index=None, geo_info=None, inp_time_info=None, tar_time_info=None):
        encoder_data = self.encoder(data, time_info=inp_time_info)
        out = self.decoder(encoder_data[-1], index, time_info=tar_time_info, geo_info=geo_info)


        return out

        