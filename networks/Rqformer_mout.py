""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
Code/weights from https://github.com/microsoft/Swin-Transformer
"""

from turtle import forward
import torch
from torch import nn, Tensor
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

class final_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output  



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


class local_decoder_layer(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.) -> None:
        super().__init__()
 
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)

        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, drop=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_pos=None, memory_pos=None):
        
        tgt1 = self.self_norm(tgt)

        q = k = self.with_pos_embed(tgt1, tgt_pos)
        tgt1 = self.self_attn(q, k, value=tgt1)[0]
        tgt = tgt + self.self_dropout(tgt1)

        tgt1 = self.cross_norm(tgt)
        tgt1 = self.cross_attn(query=self.with_pos_embed(tgt1, tgt_pos), key=self.with_pos_embed(memory, memory_pos), value=memory)[0]
        tgt = tgt + self.cross_dropout(tgt1)
        tgt = tgt + self.mlp(self.mlp_norm(tgt))

        return tgt


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
                 geo_inchans=0, patch_norm=False, local_dec_layers=1, pre_train=True):
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

        self.local_dec_layers = local_dec_layers
        self.pre_train = pre_train
        # self.pos_query = nn.Parameter(torch.zeros(query_nums, img_size[-2]*img_size[-1]))
        # nn.init.trunc_normal_(self.pos_query, std=.02)

        # self.dim_query = nn.Parameter(torch.zeros(query_nums, embed_dim))
        # nn.init.trunc_normal_(self.dim_query, std=.02)

        # self.memory_pos = nn.Parameter(torch.zeros(img_size[-2]*img_size[-1], embed_dim))
        # nn.init.trunc_normal_(self.memory_pos, std=.02)

        
        # self.pos_dec_layers = nn.ModuleList()
        # self.dim_dec_layers = nn.ModuleList()

        # for i in range(local_dec_layers):
        #     pos_local_layer = local_decoder_layer(img_size[-2]*img_size[-1], num_heads=8, qkv_bias=True, dropout=drop_rate)
        #     dim_local_layer = local_decoder_layer(embed_dim, num_heads=6, qkv_bias=True, dropout=drop_rate)
        #     self.pos_dec_layers.append(pos_local_layer)
        #     self.dim_dec_layers.append(dim_local_layer)

        self.imnet = Siren(in_features=embed_dim+3, out_features=embed_dim, hidden_features=[embed_dim, embed_dim, embed_dim*4], hidden_layers=2, outermost_linear=True)

        if use_time_info:
            self.time_embed = time_embed(embed_dim)
        else:
            self.time_embed = None


        if use_geo_info:
            self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=geo_inchans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = None

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # self.cross_layers = nn.ModuleList()
        # self.linear_layers = nn.ModuleList()

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

        # self.final = final_MLP(embed_dim, embed_dim, out_chans, 3)
        self.final = nn.Linear(embed_dim, out_chans)
        if self.pre_train:
            self.pre_enc = Siren(out_chans+2, out_features=embed_dim, hidden_features=[embed_dim, embed_dim, embed_dim*4], hidden_layers=2, outermost_linear=True)
        

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.apply(self._init_weights)
        with torch.no_grad():
            nn.init.trunc_normal_(self.final.weight, std=.02)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, memory, index=None, time_info=None, geo_info=None, target=None):
        B, H, W, D = memory.shape
        res = []
        time_embed = (index / 21).unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

        coord_highres = make_coord((H, W), flatten=False).repeat(B, 1, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()

        tgt = torch.cat([memory, coord_highres, time_embed], dim=-1)

        
        

        tgt = self.imnet(tgt)
        res.append(self.final(tgt).permute(0, 3, 1, 2))
        # tgt_pos = self.pos_query.repeat(B, 1, 1)
        # tgt_dim = self.dim_query.repeat(B, 1, 1)
        # for layer in self.pos_dec_layers:
        #     tgt_pos = layer(tgt_pos, memory.reshape(B, -1, D).permute(0, 2, 1), tgt_pos=self.pos_query, memory_pos=self.memory_pos.permute(-1, -2))  #B, 20, 32X64
        
        # for layer in self.dim_dec_layers:
        #     tgt_dim = layer(tgt_dim, memory.reshape(B, -1, D), tgt_pos=self.dim_query, memory_pos=self.memory_pos)  #B, 20,D

        # batch_indexes = np.arange(0, B)
        # index = index.squeeze(-1)
        # tgt = tgt_pos[batch_indexes, index].unsqueeze(-1) * tgt_dim[batch_indexes, index].unsqueeze(1)

        # tgt = tgt.view(B, H, W, D)


        if self.pre_train and target is not None:
            target_fea = self.pre_enc(torch.cat([target.permute(0, 2, 3, 1), coord_highres], dim=-1))

            # noise_data = torch.randn_like(target_fea, device=target_fea.device) * target_fea.detach().max()*0.1
            # target_fea = target_fea + noise_data

            fealoss = torch.mean(torch.abs(tgt - target_fea))

            res.append(self.final(target_fea).permute(0, 3, 1, 2))
            tgt = target_fea
        else:
            fealoss = 0
        
        for layer in self.layers:
            tgt = layer(tgt, memory)
            res.append(self.final(tgt).permute(0, 3, 1, 2))

        # x = self.final(tgt)
        # if len(self.window_size) == 3:
        #     res = x.permute(0, 4, 1, 2, 3)
        # elif len(self.window_size) == 2:
        #     res = x.permute(0, 3, 1, 2)
        return res, fealoss
        



class Rqformer_mout(nn.Module):
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
                geo_inchans=2,
                local_dec_layers=2,
                pre_train=True) -> None:
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
                                    geo_inchans=geo_inchans,
                                    local_dec_layers=local_dec_layers,
                                    pre_train=pre_train
                                    )


    def forward(self, data, index=None, geo_info=None, inp_time_info=None, tar_time_info=None, target=None):
        encoder_data = self.encoder(data, time_info=inp_time_info)
        out, fealoss = self.decoder(encoder_data[-1], index, time_info=tar_time_info, geo_info=geo_info, target=target)


        return out, fealoss

        