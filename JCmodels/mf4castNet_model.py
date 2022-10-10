from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from configs.param import get_args
from torch.utils.checkpoint import checkpoint_sequential

from utils.registry import MODEL_REGISTRY


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
        args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w
        # import pdb; pdb.set_trace()
        self.num_blocks = args.fno_blocks
        # print('fno_blocks', args.fno_blocks)
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        if args.fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = args.fno_softshrink

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
        args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w

        self.num_blocks = args.fno_blocks
        # print('fno_blocks', args.fno_blocks)
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

        if args.fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = args.fno_softshrink

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

# ------------------------------AFNO multiframes---------------------------------------- #



class PatchEmbed_tubelet(nn.Module):
    def __init__(self, clip_size=None, patch_size=8, patch_size_T=1, in_chans=20, embed_dim=768):
        """
        args:
            clip_size: size of clip. (T, H, W)
        """
        super().__init__()

        if clip_size is None:
            raise KeyError('img is None')
        
        assert clip_size[0]%patch_size_T == 0 and clip_size[1]%patch_size == 0 and clip_size[2]%patch_size == 0, \
        f"Clip size ({clip_size[0]}*{clip_size[1]}*{clip_size[2]}) doesn't match patch size ({patch_size_T}*{patch_size}*{patch_size})."

        self.patch_size = (patch_size_T, patch_size, patch_size)
        num_patches = (clip_size[2] // self.patch_size[2]) * (clip_size[1] // self.patch_size[1]) * (clip_size[0] // self.patch_size[0])
        self.num_patches = num_patches
        self.clip_size = clip_size

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(patch_size_T ,patch_size, patch_size), stride=(patch_size_T, patch_size, patch_size))

    def forward(self, x):
        B, T, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.clip_size[1] and W == self.clip_size[
            2] and T == self.clip_size[0], f"Input clip size ({T}*{H}*{W}) doesn't match model ({self.clip_size[0]}*{self.clip_size[1]}*{self.clip_size[2]})."
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x).flatten(2) # (B, C, T*(H*(W)))
        x = x.transpose(1, 2)
        return x #(B, L, C)

class FS_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, t=1, h=14, w=8, **kwargs):
        super().__init__()
        # args = get_args()
        # args = kwargs.get('args', None)
        filter_relax = kwargs.get('filter_relax', False)
    
        self.t = t
        self.h = h
        self.w = w
        
        self.s_norm = norm_layer(dim)
        if filter_relax:
            self.s_filter = AttnFourierNeuralOperator(dim, h=h, w=w)
        else:
            self.s_filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)
        #self.s_filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)

        self.t_norm = norm_layer(dim)
        if filter_relax:
            self.t_filter = AttnFourierNeuralOperator(dim, h=1, w=t)
        else:
            self.t_filter = AdaptiveFourierNeuralOperator(dim, h=1, w=t)
        #self.t_filter = AdaptiveFourierNeuralOperator(dim, h=1, w=t)

        self.ffn_norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.double_skip = args.double_skip

    def forward(self, x):
        """
        args:
            x: input of FS-block. (B, T*(H*(W)), D)
        """
        B, _, D = x.shape
        x = x.reshape(B*self.t, -1, D)
        residual = x
        x = self.s_norm(x)
        # import pdb; pdb.set_trace()
        x = self.s_filter(x)
        x += residual              # (BT, HW, D)

        x = x.reshape(B, self.t, -1, D).transpose(1, 2)
        x = x.reshape(-1, self.t, D)
        residual = x
        x = self.t_norm(x)
        x = self.t_filter(x)
        x += residual           # (BHW, T, D)

        x = x.reshape(B, -1, self.t, D)
        x = x.transpose(1, 2)
        x = x.reshape(B, -1, D) 
        residual = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x += residual           # (B, T*(H*(W)), D)
        return x

@MODEL_REGISTRY.register()
class AFNONet_MF(nn.Module): #model.patch_embed.proj.bias.dtype
    def __init__(self, clip_size=None, patch_size_T=1, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, mlp_ratio=4.,
                 uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0, mask_area=0., **kwargs):
        """
        FS means  Factorised self-attention which referrs to vivit model 3.
        args:
            patch_size_T: length of tube i.e. patch_size in dim of T. default: 1(equal to img patch)
        """
        super().__init__()

        if clip_size is None:
            clip_size = [1, 720, 1440]
        # import pdb; pdb.set_trace()
        #pos_embed kwargss
        self.device = kwargs.get('device', 'cpu')
        self.pos_embed_type = kwargs.get('pos_embed', 'learnableAPE')
        self.filter_relax = kwargs.get('filter_relax', False)

        self.mask_aug = True if mask_area > 0 else False
        self.mask_area = mask_area

        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed_tubelet(clip_size=clip_size, patch_size_T=patch_size_T, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.pos_embed_type == 'learnableAPE':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        elif self.pos_embed_type == 'w/oPE':
            self.pos_embed = torch.zeros(1, num_patches, embed_dim).to(self.device)
        else:
            raise NotImplementedError
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.t = clip_size[0] // patch_size_T
        self.h = clip_size[1] // patch_size
        self.w = clip_size[2] // patch_size

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([FS_Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                                           norm_layer=norm_layer, t=self.t, h=self.h, w=self.w, filter_relax=self.filter_relax) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        midle_dim = 2*embed_dim
        self.compress = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(embed_dim, 2*embed_dim, kernel_size=(self.t, 1, 1), stride=(self.t, 1, 1))),
            ('act', nn.Tanh()),
        ]))


        if patch_size == 8:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(midle_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose2d(out_chans * 16, out_chans * 4, kernel_size=(2, 2), stride=(2, 2))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.ConvTranspose2d(out_chans * 4, out_chans, kernel_size=(2, 2), stride=(2, 2))
        elif patch_size == 4:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(midle_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose2d(out_chans * 16, out_chans * 4, kernel_size=(2, 2), stride=(2, 2))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        elif patch_size == 2:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(midle_dim, out_chans * 16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.Conv2d(out_chans * 16, out_chans * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        elif patch_size == 1:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(midle_dim, out_chans * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('act1', nn.Tanh()),
                ('conv2', nn.Conv2d(out_chans * 16, out_chans*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('act2', nn.Tanh())
            ]))
            self.head = nn.Conv2d(out_chans * 4, out_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            raise NotImplementedError

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        if self.pos_embed_type == 'learnableAPE':
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
        """
        args:
            x: (B, T, C, H, W)
        """
        B = x.shape[0]
        x = self.patch_embed(x) # (B, t*h*w, D)

        # import pdb; pdb.set_trace()
        if self.mask_aug:
            mask_ratio = torch.rand(1) * self.mask_area
            mask_ratio = mask_ratio.to(device=x.device)
            mask = torch.rand((B, self.patch_embed.num_patches, 1), device=x.device)
            masked_ind = mask < mask_ratio
            visible_ind = mask >= mask_ratio
            mask[masked_ind] = 0
            mask[visible_ind] = 1
            x = x * mask

        # import pdb; pdb.set_trace()
        x += self.pos_embed
        x = self.pos_drop(x)

        if not get_args().checkpoint_activations:
            for blk in self.blocks:
                x = blk(x) # (B, t*h*w, D)
        else:
            x = checkpoint_sequential(self.blocks, 4, x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.t, self.h, self.w])
        return x

    def forward(self, x):
        """
        args:
            x: input. (B, T, C, H, W)
        """
        x = self.forward_features(x) # out: (B, D, t, h, w)
        x = self.final_dropout(x) # out: (B, D, t, h, w)
        # import pdb; pdb.set_trace()
        x = self.compress(x) # out: (B, D, 1, h, w)
        x = x.squeeze(dim=2)
        # import pdb; pdb.set_trace()
        x = self.pre_logits(x) # 
        x = self.head(x)
        return x