from turtle import forward
import torch.nn as nn
import torch
from timm.models.layers import to_2tuple
from typing import Optional
import numpy as np
from networks.utils.utils import attn_norm, window_reverse, window_partition, DropPath
from networks.utils.positional_encodings import rope2, rope3


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class ScaleOffset(nn.Module):
    def __init__(self, dim, scale=True, offset=True) -> None:
        super().__init__()
        if scale:
            self.gamma = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.gamma, std=.02)
        else:
            self.gamma = None
        if offset:
            self.beta = nn.Parameter(torch.ones(dim))
        else:
            self.beta = None
    
    def forward(self, input):
        if self.gamma is not None:
            output = input * self.gamma
        else:
            output = input
        if self.beta is not None:
            output = output + self.beta
        else:
            output = output
        
        return output




class FLASH(nn.Module):
    def __init__(self, dim, window_size, uv_bias=True, attn_drop=0., proj_drop=0., expansion_factor=2, attn_type='quad') -> None:
        super().__init__()
        self.attn_type = attn_type
        self.dim = dim
        self.window_size = window_size
        self.hidden_dim = expansion_factor * dim
        self.s = 128
        self.scale = 1. / (256.) ** 0.5 / 128.
        # self.scale = 1
        
        assert len(window_size) == 3

        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)))  # [2*Mt-1 * 2*Mh-1 * 2*Mw-1]
            
        
        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords_t = torch.arange(self.window_size[2])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t], indexing="ij"))  # [3, Mt, Mh, Mw]
        # coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw*Mt]
        # # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, Mh*Mw*Mt, Mh*Mw*Mt, Mh*Mw*Mt]
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw*Mt, Mh*Mw*Mt, Mh*Mw*Mt, 3]

        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 2] += self.window_size[2] - 1
        # relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        # relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        # relative_position_index = relative_coords.sum(-1)  # [Mh*Mw*Mt, Mh*Mw*Mt]
        # self.register_buffer("relative_position_index", relative_position_index)


        self.uv = nn.Linear(dim, 2*self.hidden_dim+self.s, bias=uv_bias)
        self.quad_q_scaleoffset = ScaleOffset(self.s)
        self.quad_k_scaleoffset = ScaleOffset(self.s)
        self.quad_attn_drop = nn.Dropout(attn_drop)

        if self.attn_type == "lin":
            self.lin_q_scaleoffset = ScaleOffset(self.s)
            self.lin_k_scaleoffset = ScaleOffset(self.s)
            # self.rope_lin = PositionalEncoding3D(dim)
            self.rope_lin = rope3((16, 32, 64), self.s)


        self.proj = nn.Linear(self.hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.softmax = nn.Softmax(dim=-1)
        self.attn_norm = attn_norm(dim=-1, method='squared_relu')

        

        # self.rope_quad = PositionalEncoding3D(self.s)
        self.rope_quad = rope3(self.window_size, self.s)


        # nn.init.normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B, T, H, W, C = x.shape
        

        x_windows = window_partition(x, self.window_size)  # [B, nW, Mt, Mh, Mw, C]
        x = x_windows.view(B, -1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # [B, nW, Mt*Mh*Mw, C]

        B_, nW, N, C = x.shape 
        # u,v:[batch_size, num_windows, Mh*Mw, hidden_dim], base:[batch_size, num_windows, Mh*Mw, s]
        u, v, base = torch.split(self.uv(x), [self.hidden_dim, self.hidden_dim, self.s], dim=-1)
        # quad_q, quad_k: [batch_size, num_windows, Mh*Mw, s]
        quad_q, quad_k = self.quad_q_scaleoffset(base), self.quad_k_scaleoffset(base)

        quad_q = self.rope_quad(quad_q.reshape(-1, *self.window_size, self.s)).reshape(B_, nW, N, self.s)
        quad_k = self.rope_quad(quad_k.reshape(-1, *self.window_size, self.s)).reshape(B_, nW, N, self.s)

        if self.attn_type == 'lin':
            lin_q, lin_k = self.lin_q_scaleoffset(base), self.lin_k_scaleoffset(base)
            lin_q = window_reverse(lin_q, self.window_size, T, H, W)
            lin_q = self.rope_lin(lin_q)
            lin_q = window_partition(lin_q, self.window_size).reshape(B, nW, N, self.s)
            lin_k = window_reverse(lin_k, self.window_size, T, H, W)
            lin_k = self.rope_lin(lin_k)
            lin_k = window_partition(lin_k, self.window_size).reshape(B, nW, N, self.s)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # quad_attn: [batch_size, num_windows, Mh*Mw, Mh*Mw]
        quad_q = quad_q * self.scale
        quad_attn = quad_q @ quad_k.transpose(-2, -1)
        # quad_attn = quad_attn*self.scale

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw] -> [Mh*Mw,Mh*Mw]
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N)
            
        # relative_position_bias = relative_position_bias.contiguous()  # [Mh*Mw, Mh*Mw]
        # quad_attn = quad_attn + relative_position_bias.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            # mask: [B, nW, Mh*Mw]
            B, nW, _ = mask.shape  # num_windows
            # attn.view: [batch_size, num_windows, Mh*Mw, Mh*Mw]
            quad_attn_mask = mask.view(B, nW, 1, -1)
            attn_mask = torch.zeros_like(quad_attn_mask, dtype=quad_q.dtype)
            attn_mask = attn_mask.masked_fill(quad_attn_mask, float("-inf"))
            quad_attn = quad_attn + attn_mask
            quad_attn = self.attn_norm(quad_attn)
        else:
            quad_attn = self.attn_norm(quad_attn)
        # quad_attn:[batch_size, num_windows, Mh*Mw, Mh*Mw]
        quad_attn = self.quad_attn_drop(quad_attn)

        # quadratic: [batch_size, num_windows, Mh*Mw, hidden_dim]
        quadratic = quad_attn @ v

        if self.attn_type == 'lin':
            if mask is not None:
                # lin_mask: [B, nW, Mh*Mw, 1]
                lin_mask = torch.logical_not(mask).unsqueeze(-1)
                # lin_v: [B, nW, Mh*Mw, hidden_dim]
                lin_v = lin_mask * v / (N * nW * self.s)
            else:
                lin_v = v  / (N * nW * self.s)
            
            # lin_kv: [B, nW, s, hidden_dim]
            lin_kv = lin_k.transpose(-2, -1) @ lin_v
            # linear: [B, nW, Mh*Mw, hidden_dim]
            linear = lin_q @ torch.sum(lin_kv, dim=-3, keepdim=True)

        # @: multiply -> [batch_size*num_windows, Mh*Mw, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        if self.attn_type == 'lin':
            x = u * (quadratic + linear)
        else:
            x = u * quadratic
        x = self.proj(x)
        x = self.proj_drop(x)

        # merge windows
        attn_windows = x.view(B, -1, self.window_size[0], self.window_size[1], self.window_size[2], C)  # [B, nW, Mt, Mh, Mw, C]
        x = window_reverse(attn_windows, self.window_size, T, H, W)  # [B, T, H, W, C]

        return x


class BiFlashBlock(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., 
                uv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm = norm_layer(dim)
        self.GAU1 = FLASH(
            dim, window_size=self.window_size, uv_bias=uv_bias,
            attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask):
        

        shortcut = x
        x = self.norm(x)
        # partition windows

        # W-MSA/SW-MSA
        x = self.GAU1(x, mask=attn_mask)  # [B, nW, Mt*Mh*Mw, C]

        # FFN
        x = shortcut + self.drop_path(x)

        return x

class BiFlashLayer(nn.Module):
    def __init__(self, dim, depth, spatial_window_size, 
                spatiotemporal_window_size, mlp_ratio=4., uv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth


        self.blocks = nn.ModuleList([
            BiFlashBlock(
                dim=dim,
                window_size=spatiotemporal_window_size if (i % 2 == 0) else spatial_window_size,
                mlp_ratio=mlp_ratio,
                uv_bias=uv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, None)
    
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, num_image_tokens, num_time_tokens, num_codebook_vectors=1024, 
                dim=768, spatial_window_size=(1, 16, 16), spatiotemporal_window_size=(16, 4, 4), 
                mlp_ratio=4, n_layers=24, dropout=0.1):
        super(BidirectionalTransformer, self).__init__()
        self.num_image_tokens = num_image_tokens
        self.num_time_tokens = num_time_tokens
        self.dim = dim
        self.mask_token_id = num_codebook_vectors
        self.num_codebook_vectors = num_codebook_vectors

        self.tok_emb = nn.Embedding(num_codebook_vectors + 1, dim)
        # self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_image_tokens, dim)), 0., 0.02)
        # self.time_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_time_tokens, dim)), 0., 0.02)
        # self.register_buffer("pos_emb", nn.init.trunc_normal_(nn.Parameter(torch.zeros(1024, args.dim)), 0., 0.02))

        self.layers = nn.ModuleList()
        for i_layer in range(n_layers):
            layers = BiFlashLayer(dim=dim,
                                        depth=2,
                                        spatial_window_size=spatial_window_size,
                                        spatiotemporal_window_size=spatiotemporal_window_size,
                                        mlp_ratio=mlp_ratio,
                                        uv_bias=True,
                                        drop=0.,
                                        attn_drop=0.,
                                        drop_path=0.1)
            self.layers.append(layers)


        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=dim, out_features=dim),
            nn.SiLU(),
            nn.LayerNorm(dim, eps=1e-7)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_image_tokens, num_codebook_vectors))
        self.ln = nn.LayerNorm(dim, eps=1e-7)
        self.drop = nn.Dropout(p=dropout)
        self.apply(weights_init)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        B, T, H, W = x.shape
        x = x.reshape(B, -1)
        # x.shape:(batch_size, length)
        # with torch.no_grad():
        token_embeddings = self.tok_emb(x)   # b, l, d      #测试下这块
        token_embeddings = token_embeddings.reshape(-1, T, H*W, self.dim)
        # assert T == self.num_time_tokens
        # time_embeddings = self.time_emb[:T, :]
        # assert H*W == self.num_image_tokens
        # position_embeddings = self.pos_emb[:H*W, :]
        # position_embeddings = self.pos_emb(x)
        # token_embwith_postime = token_embeddings + time_embeddings.unsqueeze(1) + position_embeddings
        token_embwith_postime = token_embeddings
        embed = self.drop(self.ln(token_embwith_postime.reshape(B, -1, self.dim)))
        embed = embed.reshape(B, T, H, W, -1)

        for layer in self.layers:
            embed = layer(embed)

        embed = embed.reshape(-1, H*W, self.dim)
        embed = self.Token_Prediction(embed)
        logits = torch.matmul(embed, self.tok_emb.weight[:-1, :].T).reshape(B, T, H*W, -1) + self.bias

        return logits.reshape(B, T, H, W, -1)



        #试一下mask attn模式