import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath
import torch.fft
from einops import rearrange
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer() 
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.fc2  = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) #(B, H, W, C)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, fno_blocks=4, fno_bias=True, fno_softshrink=False):
        super().__init__()

        self.hidden_size = dim
        self.num_blocks  = fno_blocks
        self.block_size  = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.scale = 0.02
        self.w1   = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1   = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2   = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2   = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.bias  = nn.Linear(self.hidden_size, self.hidden_size) if fno_bias else nn.Identity()
        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        assert x.shape[-1] == self.hidden_size, "the input tensor should be (B, *dims, C) where C is the hidden_size"
        img_size= x.shape[1:-1]
        B        = x.shape[0]
        bias    = self.bias(x) if self.bias else 0 #timer.restart(2)

        fft_dim = tuple(range(1, len(img_size)+1)) #timer.record('reshape1','filter',2)
        x = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        x       = torch.fft.rfftn(x, dim=fft_dim, norm='ortho') #timer.record('rfft2','filter',2)
        x_real  = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag  = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True) #timer.record('multiply2','filter',2)
        x_real  = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0] #timer.record('multiply3','filter',2)
        x_imag  = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1] #timer.record('multiply4','filter',2)
        #x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
        x = torch.stack([x_real, x_imag], dim=-1)
        x = torch.view_as_complex(x) #timer.record('reset','filter',2)
        x = torch.fft.irfftn(x, s=img_size, dim=fft_dim, norm='ortho') #x = x.half() #timer.record('irfft2','filter',2)
        x = x.reshape(B,*img_size, -1) #timer.record('reshape4','filter',2) ## (B, H, W, N, H) -> (B, H*W, N*H)

        return x + bias


class PartReLU_Complex(nn.Module):
    def forward(self, x):
        x_real = F.relu(x.real)
        x_imag = F.relu(x.imag)
        return torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))


class AdaptiveFourierNeuralOperatorComplex(nn.Module):
    '''
    for future implement on complex value neural network.
    Not compatible with torch.amp
    '''

    def __init__(self, dim, fno_blocks=4, fno_bias=True, fno_softshrink=False, nonlinear_activate=PartReLU_Complex()):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks  = fno_blocks
        self.block_size  = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.scale = 0.02
        self.w1    = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.cfloat))
        self.b1    = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,                  dtype=torch.cfloat))
        self.w2    = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, dtype=torch.cfloat))
        self.b2    = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,                  dtype=torch.cfloat))
        self.relu  = nonlinear_activate
        self.bias  = nn.Linear(self.hidden_size, self.hidden_size) if fno_bias else None
        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        assert x.shape[-1] == self.hidden_size, "the input tensor should be (B, *dims, C) where C is the hidden_size"
        img_size = x.shape[1:-1]
        B        = x.shape[0]
        bias    = self.bias(x) if self.bias else 0 #timer.restart(2)
        # timer.record('reshape1','filter',2)
        fft_dim = tuple(range(1, len(img_size)+1))
        x = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        x = torch.fft.rfftn(x, dim=fft_dim, norm='ortho') #timer.record('rfft2','filter',2)
        x = self.multiply(x, self.w1)+self.b1
        x = self.relu(x)
        x = self.multiply(x, self.w2)+self.b2 #x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x #<-- should be applied in complex value 
        x = torch.fft.irfftn(x, s=img_size, dim=fft_dim, norm='ortho') #x = x.half() #timer.record('irfft2','filter',2)
        x = x.reshape(B,*img_size, -1) #timer.record('reshape4','filter',2) ## (B, H, W, N, H) -> (B, H*W, N*H)
        return x + bias


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, 
                 fno_blocks=3, double_skip=False, fno_bias=False, fno_softshrink=False):
        super().__init__()
        self.norm1       = norm_layer(dim)
        self.filter      = AdaptiveFourierNeuralOperator(dim, fno_blocks=fno_blocks, fno_bias=fno_bias, fno_softshrink=fno_softshrink)
        self.drop_path   = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2       = norm_layer(dim)
        mlp_hidden_dim   = int(dim * mlp_ratio)
        self.mlp         = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        '''
        receive (B,W,H,P)
        output  (B,W,H,P)
        '''
        residual = x #timer.restart(1)
        x = self.norm1(x) #timer.record('norm1','forward_features',1)
        x = self.filter(x) #timer.record('filter','forward_features',1)
        if self.double_skip:
            x += residual
            residual = x #timer.record('residual','forward_features',1)
        x = self.norm2(x) #timer.record('norm2','forward_features',1)
        x = self.mlp(x) #timer.record('mlp','forward_features',1)
        x = self.drop_path(x) #timer.record('drop_path','forward_features',1)
        x += residual #timer.record('residual','forward_features',1)

        return x
