import torch.nn as nn
import torch
from networks.utils.utils import DropPath, PatchEmbed
import copy
from networks.utils.utils import Mlp
from functools import partial
from networks.utils.Attention import SD_attn, Cross_attn
from networks.utils.Blocks import Convnet_block, Windowattn_block




class Crossattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm

        self.normx = norm_layer(dim)
        self.normy = norm_layer(dim)
        # self.GAU1 = Flash_attn(dim, window_size=self.window_size, uv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, expansion_factor=2, attn_type='lin')
       
        self.attn = Cross_attn(dim, window_size, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    

    def forward(self, x, y):
        shortcut = x
        # partition windows

        if self.pre_norm:
            x = shortcut + self.drop_path(self.attn(self.normx(x), self.normy(y)))
        else:
            x = self.normx(shortcut + self.drop_path(self.attn(x)))

        # W-MSA/SW-MS

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))


        return x


class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        # for i in range(len(window_size)):
        #     if window_size[-i] == img_size[-i]:
        #         self.shift_size[-i] = 0

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
                attn_type="convattn"
            )
            for i in range(depth)
        ])
  


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
    
        return x




class ConvFormer3d(nn.Module):

    def __init__(self, patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False, Weather_T=1):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.Weather_T = Weather_T

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
                                )
            self.layers.append(layers)

        self.cross_blocks = nn.ModuleList()
        if self.Weather_T > 1:
            for _ in range(self.num_layers-1):
                self.cross_blocks.append(Crossattn_block(dim=embed_dim,
                                                            window_size=window_size,
                                                            num_heads=num_heads[0],
                                                            mlp_ratio=self.mlp_ratio,
                                                            qkv_bias=qkv_bias,
                                                            drop=drop_rate,
                                                            attn_drop=attn_drop_rate))

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

    def forward(self, input):
        # x: [B, C, H, W]
        B = input.shape[0]

        reses = []
        last_data = []
        for i in range(self.Weather_T):
            x = input[:,:,i]
            x, T, H, W = self.patch_embed(x)  # x:[B, H*W, C]
            x = x + self.pos_embed
            x = self.pos_drop(x)
            if len(self.window_size) == 3:
                x = x.view(B, T, H, W, -1)
            elif len(self.window_size) == 2:
                x = x.view(B, H, W, -1)

            
            for j in range(len(self.layers)):
                layer = self.layers[j]
                x= layer(x)
                if i == 0:
                    last_data.append(x)
                else:
                    if j < len(self.cross_blocks):
                        cross_data = self.cross_blocks[j](x, last_data[j])
                        x = x + cross_data
                        last_data[j] = x
                

            x = self.final(x)


            if len(self.window_size) == 3:
                res = x.permute(0, 4, 1, 2, 3)
            elif len(self.window_size) == 2:
                res = x.permute(0, 3, 1, 2)
            
            reses.append(res)

        # x = self.norm(x)  # [B, L, C]
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return reses


