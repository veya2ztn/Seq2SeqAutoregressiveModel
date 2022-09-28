from functools import partial
from collections import OrderedDict
import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.utils.checkpoint import checkpoint_sequential
from .convNd import convNd
import numpy as np
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

class Timer:
    def __init__(self,active=True):
        self.recorder=OrderedDict()
        self.real_name=OrderedDict()
        self.father=OrderedDict()
        self.child =OrderedDict()
        self.last_time={}
        self.active = active
    def restart(self,level=0):

        self.last_time[level] = time.time()
    def record(self, name,father=None,level=0):
        if not self.active:return
        cost= time.time()- self.last_time[level]

        if name not in self.recorder:self.recorder[name]=[]
        self.recorder[name].append(cost)
        if father is not None:
            if father not in self.child:self.child[father]=[]
            if name not in self.child[father]:
                self.child[father].append(name)
            self.father[name]=father
        self.real_name[name]=name
        self.last_time[level] = time.time()

    def show_stat_per_key(self,key, level=0):
        print("--"*level+f"[{self.real_name[key]}]:cost {np.mean(self.recorder[key]):.1e} ± {np.std(self.recorder[key]):.1e}")
        if key not in self.child:return
        level=level+1
        for child in self.child[key]:
            self.show_stat_per_key(child,level)

    def show_stat(self):
        if not self.active:return
        for key in self.recorder.keys():
            if key in self.father:continue
            print(">"+f"[{key}]:cost {np.mean(self.recorder[key]):.1e} ± {np.std(self.recorder[key]):.1e}")
            if key not in self.child:continue
            for child in self.child[key]:
                self.show_stat_per_key(child,1)
timer = Timer(False)

from  torch.cuda.amp import custom_fwd,custom_bwd

class BaseModel(nn.Module):

    def freeze_stratagy(self,step):
        pass

class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, img_size, fno_blocks=4,fno_bias=True, fno_softshrink=False):
        super().__init__()

        self.hidden_size = dim
        self.img_size   = img_size
        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1) if fno_bias else None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1) if self.bias else 0
        #timer.restart(2)
        x = x.reshape(B, *self.img_size, C)
        fft_dim = tuple(range(1,len(self.img_size)+1))
        #timer.record('reshape1','filter',2)
        x = torch.fft.rfftn(x, dim=fft_dim, norm='ortho');
        #timer.record('rfft2','filter',2)
        x = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        #timer.record('reshape2','filter',2)
        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        #timer.record('multiply1','filter',2)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        #timer.record('multiply2','filter',2)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        #timer.record('multiply3','filter',2)
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]
        #timer.record('multiply4','filter',2)

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        #with torch.cuda.amp.autocast(enabled=False):
        #x = x.float()
        x = torch.view_as_complex(x)
        #timer.record('reset','filter',2)
        x = x.flatten(-2,-1)
        #timer.record('reshape3','filter',2)
        x = torch.fft.irfftn(x, s=self.img_size,dim=fft_dim, norm='ortho')
        #x = x.half()
        #timer.record('irfft2','filter',2)
        x = x.reshape(B, N, C)
        #timer.record('reshape4','filter',2)
        return x + bias

class PartReLU_Complex(nn.Module):
    def forward(self,x):
        x_real = F.relu(x.real)
        x_imag = F.relu(x.imag)
        return torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

class AdaptiveFourierNeuralOperatorComplex(nn.Module):
    '''
    for future implement on complex value neural network.
    Not compatible with torch.amp
    '''
    def __init__(self, dim, img_size, fno_blocks=4,fno_bias=True, fno_softshrink=False,nonlinear_activate=PartReLU_Complex()):
        super().__init__()

        self.hidden_size = dim
        self.img_size   = img_size
        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,
                                                              self.block_size, dtype=torch.cfloat))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,
                                                              dtype=torch.cfloat))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,
                                                              self.block_size, dtype=torch.cfloat))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size,
                                                              dtype=torch.cfloat))
        self.relu = nonlinear_activate

        if fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1) if self.bias else 0
        #timer.restart(2)
        x = x.reshape(B, *self.img_size, C)
        fft_dim = tuple(range(1,len(self.img_size)+1))
        #timer.record('reshape1','filter',2)
        x = torch.fft.rfftn(x, dim=fft_dim, norm='ortho');
        #timer.record('rfft2','filter',2)
        x = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        #timer.record('reshape2','filter',2)
        x = self.multiply(x,self.w1)+self.b1
        x = self.relu(x)
        x = self.multiply(x,self.w2)+self.b2
        #x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x #<-- should be applied in complex value
        #with torch.cuda.amp.autocast(enabled=False):
        #x = x.float()
        #x = torch.view_as_complex(x)
        #timer.record('reset','filter',2)
        x = x.flatten(-2,-1)
        #timer.record('reshape3','filter',2)
        x = torch.fft.irfftn(x, s=self.img_size,dim=fft_dim, norm='ortho')
        #x = x.half()
        #timer.record('irfft2','filter',2)
        x = x.reshape(B, N, C)
        #timer.record('reshape4','filter',2)
        return x + bias

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, region_shape=(14,8), fno_blocks=3,double_skip=False, fno_bias=False, fno_softshrink=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AdaptiveFourierNeuralOperator(dim, region_shape,fno_blocks=fno_blocks,fno_bias=fno_bias,fno_softshrink=fno_softshrink)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        #timer.restart(1)
        x = self.norm1(x)
        #timer.record('norm1','forward_features',1)
        x = self.filter(x)
        #timer.record('filter','forward_features',1)
        if self.double_skip:
            x += residual
            residual = x;
        #timer.record('residual','forward_features',1)
        x = self.norm2(x)
        #timer.record('norm2','forward_features',1)
        x = self.mlp(x)
        #timer.record('mlp','forward_features',1)
        x = self.drop_path(x)
        #timer.record('drop_path','forward_features',1)
        x += residual
        #timer.record('residual','forward_features',1)
        return x

def transposeconv_engines(dim):
    return lambda *args,**kargs:convNd(*args,**kargs,num_dims=dim,is_transposed=True,use_bias=False)

def conv_engines(dim):
    return lambda *args,**kargs:convNd(*args,**kargs,num_dims=dim,is_transposed=False,use_bias=False)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768):
        super().__init__()

        if img_size is None:raise KeyError('img is None')
        patch_size   = [patch_size]*len(img_size) if isinstance(patch_size,int) else patch_size

        num_patches=1
        out_size=[]
        for i_size,p_size in zip(img_size,patch_size):
            if p_size%i_size:
                num_patches*=i_size// p_size
                out_size.append(i_size// p_size)
            else:
                raise NotImplementedError(f"the patch size ({patch_size}) cannot divide the img size {img_size}")
        self.img_size    = tuple(img_size)
        self.patch_size  = tuple(patch_size)
        self.num_patches = num_patches
        self.out_size    = tuple(out_size)
        #conv_engine = [nn.Conv1d,nn.Conv2d,nn.Conv3d]
        conv_engine = conv_engines(len(img_size))
        self.proj   = conv_engine(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, = x.shape[:2]
        inp_size = x.shape[2:]
        assert tuple(inp_size) == self.img_size, f"Input image size ({inp_size}) doesn't match model set size ({self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AFNONet(BaseModel):
    def __init__(self, img_size, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, mlp_ratio=4.,
                 uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0, checkpoint_activations=False, fno_blocks=3,double_skip=False,
                 fno_bias=False, fno_softshrink=False,debug_mode=False,history_length=1,reduce_Field_coef=False):
        super().__init__()

        assert img_size is not None
        patch_size   = [patch_size]*len(img_size) if isinstance(patch_size,int) else patch_size
        if history_length > 1:
            img_size = (history_length,*img_size)
            patch_size = (1,*patch_size)
        # print("============model:AFNONet================")
        # print(f"img_size:{img_size}")
        # print(f"patch_size:{patch_size}")
        # print(f"in_chans:{in_chans}")
        # print(f"out_chans:{out_chans}")
        # print("========================================")
        self.history_length = history_length
        self.checkpoint_activations=checkpoint_activations
        self.embed_dim   = embed_dim
        norm_layer       = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.img_size    = img_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches      = self.patch_embed.num_patches
        patch_size       = self.patch_embed.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.final_shape = self.patch_embed.out_size

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                                           drop_path=dpr[i],
                                           norm_layer=norm_layer,
                                           region_shape=self.final_shape,
                                           double_skip=double_skip,
                                           fno_blocks=fno_blocks,
                                           fno_bias=fno_bias,
                                           fno_softshrink=fno_softshrink) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # self.num_features = out_chans * img_size[0] * img_size[1]
        # self.representation_size = self.num_features * 8
        # self.pre_logits = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(embed_dim, self.representation_size)),
        #     ('act', nn.Tanh())
        # ]))
        conf_list = [{'kernel_size':[],'stride':[],'padding':[]},
                     {'kernel_size':[],'stride':[],'padding':[]},
                     {'kernel_size':[],'stride':[],'padding':[]}]
        conv_set = {8:[[2,2,0],[2,2,0],[2,2,0]],
                    4:[[2,2,0],[3,1,1],[2,2,0]],
                    2:[[3,1,1],[3,1,1],[2,2,0]],
                    1:[[3,1,1],[3,1,1],[3,1,1]],
                    3:[[3,1,1],[3,1,1],[3,3,0]],
                   }
        for patch in patch_size:
            for slot in range(len(conf_list)):
                conf_list[slot]['kernel_size'].append(conv_set[patch][slot][0])
                conf_list[slot]['stride'].append(conv_set[patch][slot][1])
                conf_list[slot]['padding'].append(conv_set[patch][slot][2])

        #transposeconv_engine = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][len(img_size)-1]
        transposeconv_engine = transposeconv_engines(len(img_size))
        self.pre_logits = nn.Sequential(OrderedDict([
            ('conv1', transposeconv_engine(embed_dim, out_chans*16, **conf_list[0])),
            ('act1', nn.Tanh()),
            ('conv2', transposeconv_engine(out_chans*16, out_chans*4, **conf_list[1])),
            ('act2', nn.Tanh())
        ]))

        # Generator head
        # self.head = nn.Linear(self.representation_size, self.num_features)
        self.head = transposeconv_engine(out_chans*4, out_chans, **conf_list[2])

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.debug_mode=debug_mode
        self.reduce_Field_coef = torch.nn.Parameter(torch.Tensor([0]),requires_grad=False)
        if reduce_Field_coef:
            self.reduce_Field_coef = torch.nn.Parameter(torch.randn(4,1,1,1)/10)
        if self.history_length >1:
            self.last_Linear_layer = nn.Linear(out_chans*self.history_length,out_chans)

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
        B = x.shape[0]
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)
        #print(torch.std_mean(x))
        if not self.checkpoint_activations:
            for blk in self.blocks:
                x = blk(x);#print(torch.std_mean(x))
        else:
            x = checkpoint_sequential(self.blocks, 4, x)

        x = self.norm(x).transpose(1, 2);
        x = torch.reshape(x, [-1, self.embed_dim, *self.final_shape])
        return x

    def get_w_resolution_pad(self,shape):
        w_now   = shape[-2]
        w_should= self.img_size[-2]
        if w_now == w_should:return None
        if w_now > w_should:
            raise NotImplementedError
        if w_now < w_should and not ((w_should - w_now)%2):
            # we only allow symmetry pad
            return (w_should - w_now)//2

    def freeze_stratagy(self,step):
        if len(self.reduce_Field_coef)>1:
            if step%2 or step==0:
                for p in self.parameters():p.requires_grad=True
                self.reduce_Field_coef.requires_grad=False
            else:
                for p in self.parameters():p.requires_grad=False
                self.reduce_Field_coef.requires_grad=True
        else:
            pass

    def forward(self, x):
        ### we assume always feed the tensor (B, p*z, h, w)
        shape = x.shape
        #print(x.shape)
        # argue the w resolution.
        pad = self.get_w_resolution_pad(shape)
        if pad is not None:
            x = F.pad(x.flatten(0,1),(0,0,pad,pad),mode='replicate').reshape(*shape[:-2],-1,shape[-1])
        #print(x.shape)
        B = x.shape[0]
        ot_shape = x.shape[2:]
        x = x.reshape(B,-1,*self.img_size)# (B, p, z, h, w) or (B, p, h, w)
        #timer.restart(level=0)
        #print(torch.std_mean(x))
        x = self.forward_features(x);#print(torch.std_mean(x))
        #timer.record('forward_features',level=0)
        x = self.final_dropout(x)
        #timer.record('final_dropout',level=0)
        x = self.pre_logits(x);#print(torch.std_mean(x))
        #timer.record('pre_logits',level=0)
        x = self.head(x);#print(torch.std_mean(x))
        if self.history_length >1:
            x = x.flatten(1,2).transpose(1,-1)
            x = self.last_Linear_layer(x)
            x = x.transpose(1,-1)
            ot_shape=ot_shape[1:]
        #timer.record('head',level=0)
        x = x.reshape(B,-1,*ot_shape)
        if pad is not None:
            x = x[...,pad:-pad,:]
        #timer.show_stat()
        #print("============================")

        return x
