from model.afnonet import BaseModel
import torch.nn as nn
import torch


from einops.layers.torch import Rearrange
from copy import deepcopy as dcp
from sympy.utilities.iterables import multiset_permutations
class ChannelShiftWide1(nn.Module):
    def __init__(self, img_size=(32,64), in_chans=70, out_chans=70,dropout_rate=0.1,**kargs):
        super().__init__()
        P=in_chans
        W,H=img_size
        blocks = [
        nn.Sequential(Rearrange('B P W H -> (B W) (P H)'),nn.Linear(P*H,P*H),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B W) (P H) -> B P W H', P=P,W=W,H=H)),
        nn.Sequential(Rearrange('B P W H -> (B H) (P W)'),nn.Linear(P*W,P*W),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B H) (P W) -> B P W H', P=P,W=W,H=H)),
        nn.Sequential(Rearrange('B P W H -> (B P) (H W)'),nn.Linear(H*W,H*W),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B P) (H W) -> B P W H', P=P,W=W,H=H))]
        self.layers=nn.ModuleList()
        for block in multiset_permutations(blocks):
            blks = [dcp(p) for p in block]
            self.layers.append(nn.Sequential(*blks))
        self.pooling = nn.Sequential(nn.BatchNorm2d(P*6),nn.Conv2d(P*6, out_chans, 3, padding=1))
    def forward(self,x):
        x = torch.cat([layer(x) for layer in self.layers],1)
        x = self.pooling(x)
        return x

class ChannelShiftDeep1(nn.Module):
    def __init__(self, img_size=(32,64), in_chans=70, out_chans=70,dropout_rate=0.1,**kargs):
        super().__init__()
        P=in_chans
        W,H=img_size
        blocks = [
        nn.Sequential(Rearrange('B P W H -> (B W) (P H)'),nn.Linear(P*H,P*H),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B W) (P H) -> B P W H', P=P,W=W,H=H)),
        nn.Sequential(Rearrange('B P W H -> (B H) (P W)'),nn.Linear(P*W,P*W),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B H) (P W) -> B P W H', P=P,W=W,H=H)),
        nn.Sequential(Rearrange('B P W H -> (B P) (H W)'),nn.Linear(H*W,H*W),nn.Dropout(p=dropout_rate),nn.Tanh(),Rearrange('(B P) (H W) -> B P W H', P=P,W=W,H=H))]
        self.layers=nn.Sequential()
        for block in multiset_permutations(blocks):
            blks = [dcp(p) for p in block]
            self.layers.append(nn.Sequential(*blks))
        self.layers.append(nn.Sequential(Rearrange('B P W H -> (B W H) P'),nn.Linear(P,P),Rearrange('(B W H) P -> B P W H', P=P,W=W,H=H)))
    def forward(self,x):
        x = self.layers(x)
        return x