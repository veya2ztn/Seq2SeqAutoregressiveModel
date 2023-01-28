import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .afnonet import AFNONet,BaseModel

class FeaturePickModel(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
    def forward(self, Field):
        return self.backbone(Field)

class OnlyPredSpeed(FeaturePickModel):
    pred_channel_for_next_stamp = list(range(28))
class WithoutSpeed(FeaturePickModel):
    pred_channel_for_next_stamp = list(range(28,70))    
class CrossSpeed(FeaturePickModel):
    train_channel_from_next_stamp = list(range(28))
    pred_channel_for_next_stamp = list(range(28,70))
class UVTP2p(FeaturePickModel):
    pred_channel_for_next_stamp = list(range(14*3,14*4-1))

    
class UVTPp2uvt(FeaturePickModel):
    train_channel_from_next_stamp = list(range(14*3,14*4-1))
    pred_channel_for_next_stamp = list(range(14*3))


class UVTP2uvt(FeaturePickModel):
    pred_channel_for_next_stamp = list(range(14*3))

class UVTp2uvt(FeaturePickModel):
    train_channel_from_this_stamp = list(range(14*3))
    train_channel_from_next_stamp = list(range(14*3,14*4-1))
    pred_channel_for_next_stamp   = list(range(14*3))

class UVTPt2uvp(FeaturePickModel):
    train_channel_from_next_stamp = list(range(14*2,14*3))
    pred_channel_for_next_stamp   = list(range(14*2)) + list(range(14*3,14*4-1))

class UVTPuv2tp(FeaturePickModel):
    train_channel_from_next_stamp = list(range(14*2))
    pred_channel_for_next_stamp   = list(range(14*2,14*4-1))

class UVTP2tp(FeaturePickModel):
    pred_channel_for_next_stamp   = list(range(14*2,14*4-1))

class TPuv2tp(FeaturePickModel):
    train_channel_from_this_stamp = list(range(14*2,14*4-1))
    train_channel_from_next_stamp = list(range(14*2))
    pred_channel_for_next_stamp   = list(range(14*2,14*4-1))


class UVTP2uvp(FeaturePickModel):
    pred_channel_for_next_stamp   = list(range(14*2)) + list(range(14*3,14*4-1))

class UVPt2uvp(FeaturePickModel):
    train_channel_from_this_stamp = list(range(14*2)) + list(range(14*3,14*4-1))
    train_channel_from_next_stamp = list(range(14*2,14*3))
    pred_channel_for_next_stamp   = list(range(14*2)) + list(range(14*3,14*4-1))

class CombM_UVTP2p2uvt(BaseModel):
    pred_channel_for_next_stamp   = list(range(55))
    def __init__(self,  args, backbone1, backbone2,ckpt1="",ckpt2=""):
        super().__init__()
        self.UVTP2p  =  UVTP2p(args,backbone1)
        ckpt1=ckpt1.strip()
        print(f"load UVTP2p model from {ckpt1}")
        if ckpt1:
            self.UVTP2p.load_state_dict(torch.load(ckpt1, map_location='cpu')['model'])
        self.UVTPp2uvt = UVTPp2uvt(args,backbone2)
        print(f"load UVTPp2uvt model from {ckpt2}")
        ckpt2=ckpt2.strip()
        if ckpt2:
            self.UVTPp2uvt.load_state_dict(torch.load(ckpt2, map_location='cpu')['model'])
    def forward(self, UVTP):
        p = self.UVTP2p(UVTP)
        uvt= self.UVTPp2uvt(torch.cat([UVTP,p],1))
        return torch.cat([uvt,p],1)

class CombM_UVTP2p2uvt_1By1(CombM_UVTP2p2uvt):
    phase = "UVTP2(p)->UVTP(p)2(u)(v)(t)"
    def enter_into_phase1(self):
        self.phase = "UVTP2(p)->UVTP(p)2(u)(v)(t)"
        self.pred_channel_for_next_stamp = list(range(55))
        for p in self.UVTP2p.parameters():p.requires_grad=False
        for p in self.UVTPp2uvt.parameters():p.requires_grad=True
    def enter_into_phase2(self):
        self.phase = "UVTPp2(u)(v)(t)->(u)(v)(t)p2[p]"
        self.pred_channel_for_next_stamp = list(range(14*3,14*4-1))
        for p in self.UVTP2p.parameters():p.requires_grad=True
        for p in self.UVTPp2uvt.parameters():p.requires_grad=False
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if epoch%2 == 0:self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()
    def forward(self, UVTP):
        B, P, L, W, H = UVTP.shape 
        assert L==2
        UVTP1=UVTP[:,:,0]
        UVTP2=UVTP[:,:,1]
        # we will inject 3 time stamp data each iter the first two will stack 
        # and achieve a (B,2,55,32,64) tensor and the last one is target
        # in "UVTP2(p)->UVTP(p)2(u)(v)(t)" mode, we fix UVTP2p and train UVTPp2uvt
        # so we only need the second time stamp data and target time stamp
        if self.phase == "UVTP2(p)->UVTP(p)2(u)(v)(t)":
            UVTP = UVTP2
            p = self.UVTP2p(UVTP)
            uvt= self.UVTPp2uvt(torch.cat([UVTP,p],1))
            return torch.cat([uvt,p],1)#(B,42,32,64)
        elif self.phase == "UVTPp2(u)(v)(t)->(u)(v)(t)p2[p]":
            p  = UVTP2[:,42:55]
            UVTP = UVTP1
            uvt = self.UVTPp2uvt(torch.cat([UVTP,p],1))
            p  = self.UVTP2p(torch.cat([uvt,p],1))
            return p #(B,13,32,64)
        
class CombM_UVTP2p2uvt_1By0(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        self.enter_into_phase1()
        
class CombM_UVTP2p2uvt_0By1(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            self.enter_into_phase2()
        else:
           self.enter_into_phase1()


class CombM_UVTP2p2uvt_2By1(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if epoch%3 in [0,1]:self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()
class CombM_UVTP2p2uvt_10By1(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if epoch%11 in list(range(10)):self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()

class CombM_UVTP2p2uvt_5By5(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if epoch%10 in list(range(5)):self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()

class CombM_UVTP2p2uvt_rand7030(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if np.random.rand()<0.7:self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()

class CombM_UVTP2p2uvt_rand9010(CombM_UVTP2p2uvt_1By1):
    def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
        if not eval_mode:
            if np.random.rand()<0.9:self.enter_into_phase1()
            else:self.enter_into_phase2()
        else:
           self.enter_into_phase1()

class CombM_UVTP2p2uvtFix(BaseModel):
    pred_channel_for_next_stamp   = list(range(55))
    def __init__(self,  args, backbone1, backbone2,ckpt1,ckpt2):
        super().__init__()
        self.UVTP2p  =  UVTP2p(args,backbone1)
        print(f"load UVTP2p model from {ckpt1}")
        if ckpt1:self.UVTP2p.load_state_dict(torch.load(ckpt1, map_location='cpu')['model'])
        #for p in self.UVTP2p.parameters():p.requires_grad=False
        self.UVTPp2uvt = UVTPp2uvt(args,backbone2)
        print(f"load UVTPp2uvt model from {ckpt2}")
        if ckpt2:self.UVTPp2uvt.load_state_dict(torch.load(ckpt2, map_location='cpu')['model'])
    def forward(self, UVTP):
        #assert not next(self.UVTP2p.parameters()).requires_grad ## use torch.no_grad is same
        with torch.no_grad():
            p = self.UVTP2p(UVTP)
        uvt= self.UVTPp2uvt(torch.cat([UVTP,p],1))
        return torch.cat([uvt,p],1)


class ReviseAFONet(BaseModel):
    feature_engine_path = "checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_09_02_17_57-seed_73001/backbone.best.pt"
    def __init__(self,  args, backbone):
        super().__init__()
        self.feature_engine  =  AFNONet(img_size=(32,64),depth=12,in_chans=70,embed_dim=768,out_chans=70,n_heads=8,patch_size=2,fno_blocks=4)
        print(f"load feature_engine model from {self.feature_engine_path}")
        self.feature_engine.load_state_dict(torch.load(self.feature_engine_path, map_location='cpu')['model'])
        self.backbone = backbone
    def forward(self, x):
        with torch.no_grad():
            y_rough = self.feature_engine(x)
        x  = torch.cat([x,y_rough],1)
        x  = self.backbone(x)
        return x


class UVTPHSC2p(FeaturePickModel):
    # make sure your dataset is 2D68N with add_ConstDirectly=True and add_LunaSolarDirectly=True
    pred_channel_for_next_stamp = list(range(14*3,14*4-1)) # pick up the potential index 
    