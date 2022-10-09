import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class SpaceLEmbedding(nn.Module):
    def __init__(self, c_in, d_model, space_num=2):
        super().__init__()
        self.space_num=space_num
        assert self.space_num > 1
        convNd = [nn.Conv1d,nn.Conv2d,nn.Conv3d][space_num-1]
        self.LConv = convNd(in_channels=c_in, out_channels=d_model,
                     kernel_size=3, padding=1, padding_mode='replicate', bias=False)
    def forward(self, x):
        # x --> [Batch,  *space_dims, in_channels] -> [Batch, h ,w, T1, in_channels]
        assert len(x.shape)==self.space_num + 3
        ### then there are at least two spatial dim 
        ### we will apply a high dimension conv as the Laplace 
        permute_order_ = [0, -2, -1] + list(range(1, self.space_num+1))
        x = x.permute(*permute_order_)#-> [Batch, T1, in_channels, h ,w]
        shape = x.shape
        x = self.LConv(x.flatten(0,1)).reshape(*shape[:2],-1,*shape[3:])#-> [Batch, T1, out_channels, h ,w]
        permute_order_ = [0] + list(range(3, self.space_num+3)) + [1,2]
        x = x.permute(*permute_order_)# x --> [Batch,  *space_dims, out_channels] -> [Batch, h ,w, T1, out_channels]
        return x

class SpaceDEmbedding(nn.Module):
    def __init__(self, c_in, d_model, space_num=2):
        super().__init__()
        assert d_model%space_num==0
        self.DConv = nn.Conv1d(in_channels=c_in, out_channels=d_model//space_num,
                         kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.space_num=space_num
    def forward(self, x):
        # x --> [Batch,  *space_dims, in_channels] -> [Batch, h ,w, T1, in_channels]
        shape = x.shape
        assert len(shape)==self.space_num + 3
        
        Dx_list = []
        for space_dim in range(1, self.space_num + 1): # the 
            Dx = x.transpose(space_dim,0)# -> [T1, h ,w, Batch, in_channels ]
            Dshape = Dx.shape
            Dx = self.DConv(Dx.flatten(1,-2).permute(1,2,0))# -> [h*w*Batch, out_channels, T1]
            Dx = Dx.permute(2,0,1).reshape(*Dshape[:-1],-1)# -> [T1, h ,w, Batch, out_channels ]
            Dx = Dx.transpose(space_dim, 0 )#-> [Batch,  h ,w, T1, out_channels]
            Dx_list.append(Dx)
        return torch.cat(Dx_list,dim=-1)

class TimeDEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                       kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        # x -> [Batch, h ,w, T1, in_channels ]
        shape    = x.shape
        BSpace_shape = shape[:-1]
        x = x.flatten(0,-3).permute(0, 2, 1)#-->(BSpace, T, C)-->(BSpace, C, T)
        x = self.tokenConv(x).permute(0, 2, 1).reshape(*BSpace_shape, -1 )
        return x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.d_inp = d_inp
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding    = TimeDEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        a = self.value_embedding(x)
        b = self.temporal_embedding(x_mark)
        reshape = False
        if a.shape[0]!= b.shape[0]:
            a = a.reshape(len(b),-1,a.shape[-2],a.shape[-1])
            b = b.unsqueeze(1)
            reshape = True

        x = a + b
        if reshape: x = x.flatten(0,1)
        return self.dropout(x)

class DataEmbedding_SLSDTD(nn.Module):
    def __init__(self, c_in, d_model, space_num, freq='h', dropout=0.1):
        super().__init__()

        self.SL_embedding  = SpaceLEmbedding(c_in=c_in, d_model=d_model,space_num=space_num)
        self.SD_embedding  = SpaceDEmbedding(c_in=c_in, d_model=d_model,space_num=space_num)
        self.TD_embedding  = TimeDEmbedding(c_in=c_in, d_model=d_model)
        self.TV_embedding  = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.TV_shape  = [1]*space_num+[-1,d_model]
        self.embedding_agg = nn.Linear(3*d_model,d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, x_mark):
        a = self.SL_embedding(x)
        b = self.SD_embedding(x)
        c = self.TD_embedding(x)
        d = self.TV_embedding(x_mark)
        x = self.embedding_agg(torch.cat([a,b,c],dim=-1)) + d.reshape(len(d),*self.TV_shape)
        return self.dropout(x)

        