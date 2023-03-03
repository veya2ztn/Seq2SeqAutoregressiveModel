from model.afnonet import BaseModel
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from copy import deepcopy as dcp
from sympy.utilities.iterables import multiset_permutations


try:
    from networks.SWD_former import SWD_former,SWDfromer
    class CK_SWDformer_3264(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (32,64),
                                depths= [4, 4, 4],
                                num_heads= [6, 6, 6],
                                Weather_T=1,only_swin=True)
        def forward(self,x):
            return self.backbone(x)
    class CK_SWDformer_64128(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (64,128),
                                depths= [2, 2, 2],
                                num_heads= [6, 6, 6],
                                Weather_T=1,only_swin=True)
        def forward(self,x):
            return self.backbone(x)
    class CK_SWDformer_64128Half(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [2, 2],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (32,64),
                                depths= [2, 2, 2],
                                num_heads= [6, 6, 6],
                                Weather_T=1,only_swin=True)
        def forward(self,x):
            return self.backbone(x)
    class CK_SWDformer_0505(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (5,5),
                                depths= [4, 4, 4],
                                num_heads= [6, 6, 6],
                                Weather_T=1,only_swin=True)
        def forward(self,x):
            return self.backbone(x)
    class CK_SWDFlowformer(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (32,64),
                                depths= [4, 4, 4],
                                num_heads= [6, 6, 6],
                                Weather_T=1,only_swin=True,use_flowatten=True,scale=100)
        def forward(self,x):
            return self.backbone(x)

    class CK_SWDFlowformerH128(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (32,64),
                                depths= [4, 4, 4],
                                num_heads= [128, 128, 128],
                                Weather_T=1,only_swin=True,use_flowatten=True)
        def forward(self,x):
            return self.backbone(x)

    class CK_SWDFlowformerH256(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = SWD_former(patch_size= [1, 1],
                                in_chans= kargs.get("in_chans",70),
                                out_chans= kargs.get("out_chans",70),
                                embed_dim= 768,
                                window_size= (32,64),
                                depths= [4, 4, 4],
                                num_heads= [256, 256, 256],
                                Weather_T=1,only_swin=True,use_flowatten=True)
        def forward(self,x):
            return self.backbone(x)

    from networks.LGNet_Rotaty import LGNet as LgNet_Rotaty
    class CK_LgNet_Rotaty(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = LgNet_Rotaty(img_size=kargs.get("img_size",(32,64)),patch_size= [1, 1],
                            in_chans= kargs.get("in_chans",70),
                            out_chans= kargs.get("out_chans",70),
                            embed_dim=kargs.get("embed_dim", 768),
                            window_size= (4,8),
                            depths= [4, 4, 4],
                            num_heads= [6, 6, 6],
                            Weather_T=1)
        def forward(self,x):
            return self.backbone(x)

    from networks.LGNet import LGNet
    class CK_LgNet(BaseModel):
        def __init__(self, *args, **kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = LGNet(img_size=kargs.get("img_size", (32, 64)), patch_size=[1, 1],
                                  in_chans=kargs.get("in_chans", 70),
                                  out_chans=kargs.get("out_chans", 70),
                                  embed_dim=kargs.get("embed_dim", 768),
                                  window_size=(4, 8),
                                  depths=[4, 4, 4],
                                  num_heads=[6, 6, 6],
                                  Weather_T=1, use_pos_embed=False)

        def forward(self, x):
            return self.backbone(x)

    from networks.LGCrossNet import LGNetCross
    class CK_LgNet_Cross(BaseModel):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = LGNetCross(img_size=kargs.get("img_size",(32,64)),patch_size= [1, 1],
                            in_chans= kargs.get("in_chans",70),
                            out_chans= kargs.get("out_chans",70),
                            embed_dim=kargs.get("embed_dim", 768),
                            window_size= (4,8),
                            depths= [4, 4, 4],
                            num_heads= [6, 6, 6],
                            Weather_T=1,use_pos_embed=False)
        def forward(self,x,cross):
            return self.backbone(x,cross)

    class CK_LgNet_138(LGNet):
        def __init__(self,*args,**kargs):
            super().__init__()
            print("this is pre-set model, we disable all config")
            self.backbone = LGNet(img_size=kargs.get("img_size",(32,64)),patch_size= [1, 1],
                            in_chans= 71,
                            out_chans= 138,
                            embed_dim=kargs.get("embed_dim", 1152),
                            window_size= (4,8),
                            depths= [4, 4, 4],
                            num_heads= [6, 6, 6],
                            Weather_T=1,use_pos_embed=True)
        def forward(self,x):
            return self.backbone(x)[:,:69]
    
except:
    class CK_SWDformer_3264(BaseModel):pass


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