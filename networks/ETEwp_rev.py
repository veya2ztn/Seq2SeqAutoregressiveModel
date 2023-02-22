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
from einops import rearrange
from networks.utils.mlp import Mlp


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        # x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, window_length, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_length = window_length
        if window_length == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif window_length == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x, T, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        if self.window_length == 3:
            assert L == T * H * W, "input feature has wrong size"
            if T > 1:
                x = x.view(B, T, H, W, C)
            else:
                x = x.view(B, H, W, C)
        elif self.window_length == 2:
            assert L == H * W, "input feature has wrong size"
            x = x.view(B, H, W, C)

        if len(x.shape) == 5:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x1 = x[:, 1::2, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x2 = x[:, 0::2, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x3 = x[:, 1::2, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x4 = x[:, 0::2, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x5 = x[:, 1::2, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x6 = x[:, 0::2, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x7 = x[:, 1::2, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # [B, H/2, W/2, 4*C]
            x = x.view(B, -1, 8 * C)  # [B, H/2*W/2, 4*C]
        elif len(x.shape) == 4:
            x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
            x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
            x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
            x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
            x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
            x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        else:
            print(x.shape)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
        


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
    
        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        # coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        if len(window_size) == 2 or window_size[0]==1:
            self.p_enc_model_sum = rope2(window_size[-2:], head_dim)
            self.window_size = window_size[-2:]
        elif len(window_size) == 3:
            self.p_enc_model_sum = rope3(window_size, head_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]

        q = self.p_enc_model_sum(q.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)
        k = self.p_enc_model_sum(k.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_attn(nn.Module):
    def __init__(self, dim, x_resolution, y_resolution, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.head_dim = dim // num_heads // self.y_resolution[0]
        self.scale = self.head_dim ** -0.5


        # if len(self.y_resolution) == 3:
        #     if self.y_resolution[0] == 1:
        #         self.y_resolution = self.y_resolution[-2:]
        #         self.x_resolution = self.x_resolution[-2:]
        #     else:
        #         if len(self.x_resolution) == 2:
        #             self.x_resolution = [1,]+self.x_resolution

        self.position_enc1 = rope2(self.x_resolution, self.head_dim)
        
        self.position_enc2 = rope2(self.y_resolution[-2:], self.head_dim)


        self.l_q = nn.Linear(self.dim, self.dim // self.y_resolution[0], bias=qkv_bias)
        self.l_kv = nn.Linear(self.dim, self.dim * 2 // self.y_resolution[0], bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.l_proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        B, H_x, W_x, C = x.shape
        # B, T_y, H_y, W_y, C = y.shape

        res_x = []
        for i in range(self.y_resolution[0]):
            y_ = y[:,i]

            q = self.l_q(x).reshape(B, H_x * W_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.position_enc1(q.reshape(-1, *self.x_resolution, self.head_dim)).reshape(B, self.num_heads, -1, self.head_dim)

            kv = self.l_kv(y_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1] 

            k = self.position_enc2(k.reshape(-1, *self.y_resolution[-2:], self.head_dim)).reshape(B, self.num_heads, -1, self.head_dim)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x_ = (attn @ v).transpose(1, 2).reshape(B, H_x, W_x, self.dim // self.y_resolution[0])
            res_x.append(x_)

        x = torch.cat(res_x, dim=-1)

        x = self.l_proj(x)
        x = self.proj_drop(x)
        return x









class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=[4, 8], shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        T, H, W = self.T, self.H, self.W
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, C)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        # pad_l = pad_t = 0
        # pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        # pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # _, Hp, Wp, _ = x.shape

        window_total_size = 1
        for i in self.window_size:
            window_total_size *= i

        # # cyclic shift
        # if self.shift_size[0] > 0 and L != window_total_size:
        if len(self.window_size) == 3:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        elif len(self.window_size) == 2:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    # else:
        #     shifted_x = x
        #     attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        if len(self.window_size) == 3:
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # [nW*B, Mh*Mw, C]
        elif len(self.window_size) == 2:
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        if len(self.window_size) == 3:
            attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)  # [nW*B, Mh, Mw, C]
        elif len(self.window_size) == 2:
            attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, T, H, W)  # [B, H', W', C]

        # reverse cyclic shift
        # if self.shift_size[0] > 0 and L != window_total_size:
        if len(self.window_size) == 3:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        elif len(self.window_size) == 2:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        # else:
        #     x = shifted_x

        # if pad_r > 0 or pad_b > 0:
        #     # 把前面pad的数据移除掉
        #     x = x[:, :H, :W, :].contiguous()

        x = x.view(B, T * H * W, C)

        # FFN
        x = self.norm1(shortcut + self.drop_path(x))
        x = self.norm2(x + self.drop_path(self.mlp(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, img_size, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = [i//2 for i in window_size]
        for i in range(len(window_size)):
            if window_size[-i] >= img_size[-i]:
                self.shift_size[-i] = 0
                window_size[-i] = img_size[-i]

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim // 2, window_length=len(window_size) if window_size[0]!=1 else len(window_size)-1, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, T, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        # Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        # Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        
        if len(self.window_size) == 3:
            img_mask = torch.zeros((1, T, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
            t_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], 0),
                        slice(0, None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1
        elif len(self.window_size) == 2:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], 0),
                        slice(0, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        if len(self.window_size) == 3:
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])  # [nW, Mh*Mw]
        elif len(self.window_size) == 2:
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])  # [nW, Mh*Mw]
            
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, T, H, W):
        if self.downsample is not None:
            x = self.downsample(x, T, H, W)
            T, H, W = (T + 1) // 2, (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, T, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.T, blk.H, blk.W = T, H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        

        return x, T, H, W


class SwinTransformer(nn.Module):
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

    def __init__(self, img_size, patch_size=(1, 1, 1), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        total_pix_len = 1
        for i in img_size:
            total_pix_len *= i

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_pix_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的

            for i in range(len(window_size)):
                if window_size[-i] > img_size[-i]:
                    window_size[-i] = img_size[-i]

            layers = BasicLayer(img_size=copy.deepcopy(img_size),
                                dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=copy.deepcopy(window_size),
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer > 0) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

            img_size = [(i+1)//2 for i in img_size]

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

    def forward(self, x):
        # x: [B, L, C]
        x, T, H, W = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        res = []

        for layer in self.layers:
            x, T, H, W = layer(x, T, H, W)
            B, L, C = x.shape
            if len(self.window_size) == 3:
                res.append(x.view(B, T, H, W, C))
            else:
                res.append(x.view(B, H, W, C))

        # x = self.norm(x)  # [B, L, C]
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return res





class Cross_Block(nn.Module):

    def __init__(self, img_size, dim, num_heads, window_size=[4,8], shift_size=0,
                mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.ReLU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        for i in range(len(self.img_size)):
            if self.img_size[i] == 0:
                self.img_size[i] = 1
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = Cross_attn(
            dim, x_resolution=window_size, y_resolution=img_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path[0]) if drop_path[0] > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):

        shortcut = x
        B, H, W, C = x.shape
        # x = self.norm1(x)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        # pad_l = pad_t = 0
        # pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        # pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # _, Hp, Wp, _ = x.shape

        window_total_size = 1
        for i in self.window_size:
            window_total_size *= i


        # W-MSA/SW-MSA
        x = self.attn(x, y)  # [nW*B, Mh*Mw, C]


        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinDecoder(nn.Module):
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

    def __init__(self, img_size, out_chans=20,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        

        self.random_query = nn.init.trunc_normal_(nn.Parameter(torch.zeros(*img_size[-2:], embed_dim)), 0., 0.02)

        # split image into non-overlapping patches

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的

            cross_layer = Cross_Block(img_size=[(i)//(2**i_layer) for i in img_size],
                                    dim=int(embed_dim * 2 ** i_layer),
                                    num_heads=num_heads[i_layer],
                                    window_size=img_size[-2:],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
                                    )
            layers = BasicLayer(img_size=copy.deepcopy(img_size),
                                dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size[-2:],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=None,
                                use_checkpoint=use_checkpoint)
            out_channels = embed_dim * 2 ** (i_layer+1) if i_layer < self.num_layers-1 else out_chans
            linear_layer = nn.Linear(embed_dim * 2 ** i_layer, out_channels)


            self.layers.append(layers)
            self.cross_layers.append(cross_layer)
            self.linear_layers.append(linear_layer)


        

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

    def forward(self, x):
        B = x[0].shape[0]
        res = self.random_query.repeat(B, 1, 1, 1)
        B, H, W, C = res.shape
        for i in range(len(self.layers)):
            res = self.cross_layers[i](res, x[i])
            B, H, W, C = res.shape
            res = res.reshape(B, -1, C)
            res, _, _, _ = self.layers[i](res, 1, *self.img_size[-2:])
            res = self.linear_layers[i](res)
            res = res.reshape(B, H, W, res.shape[-1])
        
        return res.permute(0, 3, 1, 2)



class ETEwp_rev(nn.Module):
    def __init__(self, img_size=[32,64], patch_size=(1,1,1), in_chans=20, out_chans=20, embed_dim=96, window_size=[4,8], depths=[2, 2, 6, 2], \
                num_heads=[3, 6, 12, 24], Weather_T=16, drop_path_rate=0.) -> None:
        super().__init__()
        img_size = [Weather_T,] + img_size
        self.encoder = SwinTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                                        embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
                                        window_size=copy.deepcopy(window_size), drop_path_rate=drop_path_rate)
        self.decoder = SwinDecoder(img_size=img_size, out_chans=out_chans, embed_dim=embed_dim, 
                                    depths=depths, num_heads=num_heads, window_size=copy.deepcopy(window_size),
                                    drop_path_rate=drop_path_rate)


    def forward(self, data):
        if len(data.shape) == 4:
            data = data.unsqueeze(2)
        encoder_data = self.encoder(data)
        out = self.decoder(encoder_data)


        return out

        