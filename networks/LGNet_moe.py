import torch.nn as nn
import torch
from networks.utils.utils import DropPath, PatchEmbed
import copy
from networks.utils.utils import Mlp
from functools import partial
from networks.utils.Attention import SD_attn
from networks.utils.Blocks import Convnet_block, Windowattn_block
from networks.utils.Blocks import Windowattn_block_withmoe
from einops import rearrange


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


class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, num_experts=1,
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, layer_type="convnet_block",
                attr_len=770, attr_hidden_size=768, attn_use_moe_list=[True, True, True, True], 
                mlp_use_moe_list=[True, True, True, True], expert_capacity=1.0, 
                router_bias=True, router_noise=1e-2, is_scale_prob=True, drop_tokens=True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        # for i in range(len(window_size)):
        #     if window_size[-i] == img_size[-i]:
        #         self.shift_size[-i] = 0
        if layer_type == "convnet_block":
            raise NotImplementedError('moe convnet block')
        elif layer_type == "window_block":
            self.blocks = nn.ModuleList([
            Windowattn_block_withmoe(
                dim=dim,
                attr_len=attr_len,
                window_size=window_size,
                attr_hidden_size=attr_hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_use_moe=attn_use_moe_list[i], 
                mlp_use_moe=mlp_use_moe_list[i], 
                num_experts=num_experts, 
                expert_capacity=expert_capacity, 
                router_bias=router_bias, 
                router_noise=router_noise, 
                is_scale_prob=is_scale_prob, 
                drop_tokens=drop_tokens
            )
            for i in range(depth)
        ])
        elif layer_type == "swin_block":
            self.blocks = nn.ModuleList([
            Windowattn_block_withmoe(
                dim=dim,
                attr_len=attr_len,
                window_size=window_size,
                attr_hidden_size=attr_hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_use_moe=attn_use_moe_list[i], 
                mlp_use_moe=mlp_use_moe_list[i], 
                num_experts=num_experts, 
                expert_capacity=expert_capacity, 
                router_bias=router_bias, 
                router_noise=router_noise, 
                is_scale_prob=is_scale_prob, 
                drop_tokens=drop_tokens,
                shift_size=[0,0] if i%2==0 else [i//2 for i in window_size]
            )
            for i in range(depth)
        ])


    def forward(self, x, attr=None):
        z_loss_list = []
        balance_loss_list = []
        for blk in self.blocks:
            x, z_loss, balance_loss = blk(x, attr)
            z_loss_list = z_loss_list + z_loss
            balance_loss_list = balance_loss_list + balance_loss
    
        return x, z_loss_list, balance_loss_list







class LGNet_moe(nn.Module):

    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 attr_len=770, attr_hidden_size=768, attn_use_moe_list=[True, True, True, True], 
                 mlp_use_moe_list=[True, True, True, True], num_experts_list=[32, 32, 32, 32], 
                 expert_capacity=1., router_bias=True, router_noise=1e-2, is_scale_prob=True, drop_tokens=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,
                 attr_type=["ind", "pos", "inp_time", "tar_time", "geo"], **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.attr_len = attr_len
        self.attr_type = attr_type

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
                            num_experts=num_experts_list[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=[img_size[-2]//patch_size[-2], img_size[-1]//patch_size[-1]] if i_layer==0 else window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            norm_layer=norm_layer,
                            layer_type="window_block" if i_layer==0 else "swin_block",
                            attr_len=attr_len, 
                            attr_hidden_size=attr_hidden_size, 
                            attn_use_moe_list=attn_use_moe_list, 
                            mlp_use_moe_list=mlp_use_moe_list,
                            expert_capacity=expert_capacity, 
                            router_bias=router_bias, 
                            router_noise=router_noise, 
                            is_scale_prob=is_scale_prob, 
                            drop_tokens=drop_tokens
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

    def forward(self, x, index=None, geo_info=None, inp_time_info=None, tar_time_info=None):
        # x: [B, C, H, W]
        z_loss_list = []
        balance_loss_list = []
        B = x.shape[0]
        x, T, H, W = self.patch_embed(x)  # x:[B, H*W, C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if len(self.window_size) == 3:
            x = x.view(B, T, H, W, -1)
        elif len(self.window_size) == 2:
            x = x.view(B, H, W, -1)


        time_embed = (index / 21).unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

        coord_highres = make_coord((H, W), flatten=False).repeat(B, 1, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()

        if self.attr_len < self.embed_dim and (self.attr_len == 1) or (self.attr_len > self.embed_dim and (self.attr_len-self.embed_dim == 1)):
            attr = time_embed
        elif self.attr_len < self.embed_dim and (self.attr_len == 2) or (self.attr_len > self.embed_dim and (self.attr_len-self.embed_dim == 2)):
            attr = coord_highres
        elif self.attr_len < self.embed_dim and (self.attr_len == 3) or (self.attr_len > self.embed_dim and (self.attr_len-self.embed_dim == 3)):
            attr = torch.cat((time_embed, coord_highres), dim=-1)
        elif self.attr_len == self.embed_dim:
            attr = time_embed[0:0]
        elif self.attr_len == 0:
            attr = None
        else:
            raise NotImplementedError("error")
        for layer in self.layers:
            x, z_loss, balance_loss = layer(x, attr)
            z_loss_list = z_loss_list + z_loss
            balance_loss_list = balance_loss_list + balance_loss

        x = self.final(x)

        res = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[-2],
            p2=self.patch_size[-1],
            h=self.img_size[0] // self.patch_size[-2],
            w=self.img_size[1] // self.patch_size[-1],
        )

        return res, z_loss_list, balance_loss_list
