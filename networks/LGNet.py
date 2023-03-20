import torch.nn as nn
import torch
import copy

from functools import partial
from networks.utils.Attention import SD_attn,SD_attn_Cross
from networks.utils.Blocks import Convnet_block, Windowattn_block, Windowattn_Cross_block
from networks.utils.utils import DropPath, PatchEmbed
from networks.utils.utils import Mlp
from einops import rearrange
#from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from networks.utils.Attention import SD_attn
from networks.utils.utils import Mlp
from networks.utils.utils import PatchEmbed


def grow_up_layernorm(child, expand):
    old_dim = child.normalized_shape[0]
    target_dim = expand*old_dim
    print(f"growing LayerNorm from ({old_dim},) to ({target_dim},)")
    new_norm = nn.LayerNorm(target_dim, child.eps, child.elementwise_affine)
    old_state_dict = child.state_dict()
    weight = old_state_dict['weight']
    bias = old_state_dict['bias']
    new_state_dict = {'weight': torch.cat(
        [weight]*expand), 'bias': torch.cat([bias]*expand)}
    new_norm.load_state_dict(new_state_dict)
    return new_norm


def grow_up_patch_embed(child, expand):
    target_dim = child.embed_dim*expand
    print(f"growing PatchEmbed from ({child.embed_dim},) to ({target_dim},)")
    assert not hasattr(child.norm, 'weight')
    new_norm = PatchEmbed(patch_size=child.patch_size,
                          in_c=child.in_chans, embed_dim=target_dim, norm_layer=None)
    old_state_dict = child.state_dict()
    weight = old_state_dict['proj.weight']
    bias = old_state_dict['proj.bias']
    new_state_dict = {'proj.weight': torch.cat([weight]*expand),
                      'proj.bias': torch.cat([bias]*expand)}
    new_norm.load_state_dict(new_state_dict)
    return new_norm


def grow_up_SingleLinear(child, expand):
    target_dim = child.in_features*expand
    print(
        f"growing final Linear from ({child.in_features},{child.out_features}) to ({target_dim},{child.out_features})")
    assert child.bias is None
    new_norm = nn.Linear(target_dim, child.out_features, bias=False)
    old_state_dict = child.state_dict()
    weight = old_state_dict['weight']
    new_state_dict = {'weight': torch.cat([weight]*expand, -1)/expand  # (68,192)->(68,192*2)
                      }
    new_norm.load_state_dict(new_state_dict)
    return new_norm

from networks.utils.Attention import SD_attn
def SD_attn_grow_up_inner_recursively(module: nn.Module, target_dim: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, SD_attn):
            setattr(module, name, child.grow_up_to(target_dim, only_inner=True))
        else:
            SD_attn_grow_up_inner_recursively(child, target_dim)

def LGnet_grow_up_full_recursively(module: nn.Module, expand: int, offset=0) -> None:
    """
    ```
        data_kargs= {'img_size': (32, 64),'patch_size': 2, 'patch_range': 5, 'in_chans': 68, 'out_chans': 68,
        'fno_blocks': 4, 'embed_dim': 48, 'depth': 12, 'debug_mode': 0, 'double_skip': False,
        'fno_bias': False, 'fno_softshrink': 0.0, 'history_length': 1, 'reduce_Field_coef': False,
        'modes': (17, 33, 6), 'mode_select': 'normal', 'physics_num': 4, 'pred_len': 1,
        'n_heads': 8, 
                    'label_len': 3, 'canonical_fft': 1, 'unique_up_sample_channel': 0,
        'share_memory': 1, 'dropout_rate': 0, 'conv_simple': 1, 'graphflag': 'mesh5', 'agg_way': 'mean'}
        model = CK_LgNet(**data_kargs)

        model1 = copy.deepcopy(model)
        model2 = copy.deepcopy(model)
        SD_attn_grow_up_full_recursively(model2,2)
        a = torch.randn(1,68,32,64).cuda()
        torch.dist(model1(a),
                model2(a),
                )
    ```
    """
    for name, child in module.named_children():
        print(" "*offset, end="")
        if isinstance(child, SD_attn):
            target_dim = expand*child.dim
            setattr(module, name, child.grow_up_to(
                target_dim, only_inner=False))
        elif isinstance(child, nn.LayerNorm):
            setattr(module, name, grow_up_layernorm(child, expand))
        elif isinstance(child, Mlp):
            setattr(module, name, child.grow_up_to(expand, only_inner=False))
        elif isinstance(child, PatchEmbed):
            setattr(module, name, grow_up_patch_embed(child, expand))
        elif isinstance(child, nn.Linear):
            setattr(module, name, grow_up_SingleLinear(child, expand))
        else:
            LGnet_grow_up_full_recursively(child, expand, offset=offset+1)

class Layer(nn.Module):
    def __init__(self, dim, depth, window_size, 
                num_heads=1, mlp_ratio=4., qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, layer_type="convnet_block",
                use_checkpoint=False,expand=1) -> None:
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
                    norm_layer=norm_layer, expand=expand
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
                    shift_size=[0,0] if i%2==0 else [i//2 for i in window_size],expand=expand
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

class LG_net(nn.Module):    

    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=3, out_chans=20,
                 embed_dim=768, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 4, 8), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=False,use_checkpoint=False,
                 use_pos_embed=True,expand=1):
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
                     window_size=[img_size[-2]//patch_size[-2], img_size[-1]//patch_size[-1]] if i_layer==0 else window_size,
                     mlp_ratio=self.mlp_ratio,
                     qkv_bias=qkv_bias,
                     drop=drop_rate,
                     attn_drop=attn_drop_rate,
                     drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                     norm_layer=norm_layer,
                     layer_type="window_block" if i_layer==0 else "swin_block",
                     use_checkpoint=use_checkpoint,expand=expand
                                )
            self.layers.append(layers)

        # self.final = nn.Linear(embed_dim, out_chans, bias=False)
        self.final = nn.Linear(embed_dim, out_chans*patch_size[-1]*patch_size[-2], bias=False)
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0]//patch_size[-2]*img_size[1]//patch_size[-1], embed_dim)) 
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = 0

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

        res = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[-2],
            p2=self.patch_size[-1],
            h =self.img_size[0] // self.patch_size[-2],
            w =self.img_size[1] // self.patch_size[-1],
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

class LGNet(nn.Module): 
    def __init__(self, img_size=[32, 64], patch_size=(1, 1, 1), in_chans=20, out_chans=20, embed_dim=768, window_size=[4, 8], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], Weather_T=16, drop_rate=0., attn_drop_rate=0., drop_path=0., use_checkpoint=False, 
                 use_pos_embed=True,expand=1) -> None:
        super().__init__()
        self.net = LG_net(img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans,
                            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                            window_size=window_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path, use_checkpoint=use_checkpoint, 
                          use_pos_embed=use_pos_embed, expand=expand)
   
    def forward(self, data, **kwargs):
        out = self.net(data)
        return out


