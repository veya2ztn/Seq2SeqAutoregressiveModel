from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft

from torch.utils.checkpoint import checkpoint_sequential



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        #args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w
        # import pdb; pdb.set_trace()
        self.num_blocks = 4
        # print('fno_blocks', 4)
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        # if args.fno_bias:
        #     self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        # else:
        self.bias = None

        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        # import pdb; pdb.set_trace()
        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C) #x_eg=torch.tensor([0, 1, 2, 3]).to(x.device) fft_eg=torch.fft.fft(x_eg) 
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0],
                        inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1],
                        inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias


class AttnFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        #args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w
        # import pdb; pdb.set_trace()
        self.num_blocks = 4
        # print('fno_blocks', 4)
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        real = torch.ones((self.h, (self.w//2)+1, self.num_blocks))
        imag = torch.zeros((self.h, (self.w//2)+1, self.num_blocks))
        self.relax_coefficients = torch.nn.Parameter(torch.stack([real, imag], dim=-1))

        # if args.fno_bias:
        #     self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        # else:
        self.bias = None

        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C) #x_eg=torch.tensor([0, 1, 2, 3]).to(x.device) fft_eg=torch.fft.fft(x_eg) 
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0],
                        inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1],
                        inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1] #B, self.h, self.w//2+1, self.blocks, self.block_dims

        #import pdb; pdb.set_trace() #torch.cuda.memory_reserved()/(2**30) torch.cuda.empty_cache()
        x_real = self.relax_coefficients[None, :, :, :, None, 0] * x_real - self.relax_coefficients[None, :, :, :, None, 1] * x_imag
        x_imag = self.relax_coefficients[None, :, :, :, None, 0] * x_imag + self.relax_coefficients[None, :, :, :, None, 1] * x_real
        
        x = torch.stack([x_real, x_imag], dim=-1) #B, self.h, self.w//2+1, self.blocks, self.block_dims, 2 
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, **kwargs):
        super().__init__()
        #args = get_args()
        self.norm1 = norm_layer(dim)
        filter_relax = kwargs.get('filter_relax', False)
        if filter_relax:
            self.filter = AttnFourierNeuralOperator(dim, h=h, w=w)
        else:
            self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = False

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x += residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x += residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768):
        super().__init__()

        if img_size is None:
            raise KeyError('img is None')

        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class AFNONet(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, fno_blocks=4, fno_bias=False,
                 mlp_ratio=4., uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0, **kwargs):
        super().__init__()

        if img_size is None:
            img_size = [720, 1440]
        self.img_size  = img_size
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        # import pdb;pdb.set_trace()
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                                           norm_layer=norm_layer, h=self.h, w=self.w) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if patch_size == 8:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(embed_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose2d(out_chans * 16, out_chans * 4, kernel_size=(2, 2), stride=(2, 2))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.ConvTranspose2d(out_chans * 4, out_chans, kernel_size=(2, 2), stride=(2, 2))
        elif patch_size == 4:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(embed_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose2d(out_chans * 16, out_chans * 4, kernel_size=(2, 2), stride=(2, 2))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        elif patch_size == 2:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(embed_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.Conv2d(out_chans * 16, out_chans * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        ##### normal ####
        elif patch_size == 1:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(embed_dim, out_chans * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect')),
                ('act1', nn.Tanh()),
                ('conv2', nn.Conv2d(out_chans * 16, out_chans*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect')),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect')

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        shape = x.shape
        #print(x.shape)
        # argue the w resolution.
        

        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        x = self.norm(x).transpose(1, 2)
        # x = self.norm(x)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        # print(x.shape)

        return x
 
    def forward(self, x):
        B = x.shape[0]
        ot_shape = x.shape[2:]
        x = x.reshape(B,-1,*self.img_size)# (B, p, z, h, w) or (B, p, h, w)

        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.pre_logits(x)
        x = self.head(x)

        x = x.reshape(B,-1,*ot_shape)
        return x



