from turtle import forward
import torch.nn as nn
import torch
from timm.models.layers import to_2tuple
from typing import Optional
from networks.utils.utils import Mlp, window_partition, window_reverse, DropPath



def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)




class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        assert len(window_size) == 3

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # [2*Mt-1 * 2*Mh-1 * 2*Mw-1, nH]
            
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t], indexing="ij"))  # [3, Mt, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw*Mt]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, Mh*Mw*Mt, Mh*Mw*Mt, Mh*Mw*Mt]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw*Mt, Mh*Mw*Mt, Mh*Mw*Mt, 3]

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw*Mt, Mh*Mw*Mt]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
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
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
            
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

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


class BiTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        B, T, H, W, C = x.shape

        shortcut = x

        # partition windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, Mt, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # [nW*B, Mt*Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mt*Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)  # [nW*B, Mt, Mh, Mw, C]
        x = window_reverse(attn_windows, self.window_size, T, H, W)  # [B, H', W', C]

        # FFN
        x = self.norm1(shortcut + self.drop_path(x))
        x = self.norm2(x + self.drop_path(self.mlp(x)))

        return x




class BiTransformerLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, spatial_window_size, 
                spatiotemporal_window_size, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth


        self.blocks = nn.ModuleList([
            BiTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=spatiotemporal_window_size if (i % 2 == 0) else spatial_window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
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
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_image_tokens, dim)), 0., 0.02)
        self.time_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_time_tokens, dim)), 0., 0.02)
        # self.register_buffer("pos_emb", nn.init.trunc_normal_(nn.Parameter(torch.zeros(1024, args.dim)), 0., 0.02))

        self.layers = nn.ModuleList()
        for i_layer in range(n_layers):
            layers = BiTransformerLayer(dim=dim,
                                        depth=2,
                                        num_heads=4,
                                        spatial_window_size=spatial_window_size,
                                        spatiotemporal_window_size=spatiotemporal_window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=True,
                                        drop=0.,
                                        attn_drop=0.,
                                        drop_path=0.1)
            self.layers.append(layers)


        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_image_tokens, num_codebook_vectors))
        self.ln = nn.LayerNorm(dim, eps=1e-12)
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
        assert T == self.num_time_tokens
        time_embeddings = self.time_emb[:T, :]
        assert H*W == self.num_image_tokens
        position_embeddings = self.pos_emb[:H*W, :]
        # position_embeddings = self.pos_emb(x)
        token_embwith_postime = token_embeddings + time_embeddings.unsqueeze(1) + position_embeddings
        embed = self.drop(self.ln(token_embwith_postime.reshape(B, -1, self.dim)))
        embed = embed.reshape(B, T, H, W, -1)

        for layer in self.layers:
            embed = layer(embed)

        embed = embed.reshape(-1, H*W, self.dim)
        embed = self.Token_Prediction(embed)
        logits = torch.matmul(embed, self.tok_emb.weight[:-1, :].T).reshape(B, T, H*W, -1) + self.bias

        return logits.reshape(B, T, H, W, -1)



        #试一下mask attn模式