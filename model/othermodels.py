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

    class CK_LgNet_138(BaseModel):
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
    
    from einops import rearrange    
    class DecodeEncodeLgNet(LGNet):
        def __init__(self, *args, **kargs):
            super().__init__(img_size=kargs.get("img_size", (32, 64)), patch_size=[1, 1],
                    in_chans=kargs.get("in_chans", 70),
                    out_chans=kargs.get("out_chans", 70),
                    embed_dim=kargs.get("embed_dim", 768),
                    window_size=(4, 8),
                    depths=[4, 4, 4],
                    num_heads=[6, 6, 6],
                    Weather_T=1, use_pos_embed=False)
        
        def encode(self, x):
            assert len(self.net.layers)>1
            # x: [B, C, H, W]
            B = x.shape[0]
            x, T, H, W = self.net.patch_embed(x)  # x:[B, H*W, C]
            x = x + self.net.pos_embed
            x = self.net.pos_drop(x)
            if len(self.net.window_size) == 3:
                x = x.view(B, T, H, W, -1)
            elif len(self.net.window_size) == 2:
                x = x.view(B, H, W, -1)

            for layer in self.net.layers[:-1]:
                x = layer(x)
            return x
        
        def decode(self,x):
            assert len(self.net.layers)>1
            x = self.net.layers[-1](x)
            x = self.net.final(x)
            x = rearrange(x, "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                            p1=self.net.patch_size[-2],
                            p2=self.net.patch_size[-1],
                            h=self.net.img_size[0] // self.net.patch_size[-2],
                            w=self.net.img_size[1] // self.net.patch_size[-1],
                          )
            return x
        
        def forward(self,x):
            return self.decode(self.encode(x))

        
    import copy
    class LgNet_MultiBranch(nn.Module):
        '''
            use it as wrapper model.
        '''
        
        def __init__(self, args, backbone):
            super().__init__()
            self.backbone = backbone
            self.new_branch= nn.ModuleList()
            assert isinstance(args.multibranch_select,(list,tuple))
            self.multibranch_select = args.multibranch_select
            # notice the first branch should be the pretrain model branch.
            for _ in args.multibranch_select[1:]:
                self.new_branch.append(
                    nn.Sequential(
                    copy.deepcopy(self.backbone.net.layers[-1]),
                    copy.deepcopy(self.backbone.net.final),
                    Rearrange("b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                            p1=self.backbone.net.patch_size[-2],
                            p2=self.backbone.net.patch_size[-1],
                            h=self.backbone.net.img_size[0] // self.backbone.net.patch_size[-2],
                            w=self.backbone.net.img_size[1] // self.backbone.net.patch_size[-1],)
                    )
                )
        def decode(self,x,branch_flag):
            try:
                control_flag = self.multibranch_select.index(branch_flag)
            except:
                print(f"multibranch_select.index is {self.multibranch_select}")
                print(f"branch_flag is {branch_flag}")
                raise
            if control_flag == 0 :
                return self.backbone.decode(x)
            else:
                return self.new_branch[control_flag-1](x)

        def forward(self, x, branch_flag):
            assert isinstance(branch_flag,int)
            return self.decode(self.backbone.encode(x), branch_flag)
    
    class LgNet_MultiHead(nn.Module):
        '''
            use it as wrapper model.
            base branch is 6;
        '''
        branch1_weight_path  = "checkpoints/WeathBench32x64/LgNet_Head/ts_2_finetune-2D706N_per_1_step/03_10_19_52_46121-seed_73001/pretrain_latest.pt"
        branch3_weight_path  = "checkpoints/WeathBench32x64/LgNet_Head/ts_2_finetune-2D706N_per_24_step/03_10_19_38_58203-seed_73001/pretrain_latest.pt"
        branch24_weight_path = "checkpoints/WeathBench32x64/LgNet_Head/ts_2_finetune-2D706N_per_3_step/03_11_00_09_47151-seed_73001/pretrain_latest.pt"
        def __init__(self, args, backbone):
            super().__init__()
            self.backbone = backbone
            rearrange_layer= Rearrange("b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",p1=self.backbone.net.patch_size[-2],
                                                    p2=self.backbone.net.patch_size[-1],
                                                    h=self.backbone.net.img_size[0] // self.backbone.net.patch_size[-2],
                                                    w=self.backbone.net.img_size[1] // self.backbone.net.patch_size[-1],)
            self.branch1  = nn.Sequential(copy.deepcopy(self.backbone.net.layers[-1]),copy.deepcopy(self.backbone.net.final),rearrange_layer)
            layer_weight, final_weight = self.distiall_head_weight(self.branch1_weight_path)
            self.branch1[0].load_state_dict(layer_weight)
            self.branch1[1].load_state_dict(final_weight)

            self.branch24  = nn.Sequential(copy.deepcopy(self.backbone.net.layers[-1]),copy.deepcopy(self.backbone.net.final),rearrange_layer)
            layer_weight, final_weight = self.distiall_head_weight(self.branch24_weight_path)
            self.branch24[0].load_state_dict(layer_weight)
            self.branch24[1].load_state_dict(final_weight)

            self.branch3 = nn.Sequential(copy.deepcopy(self.backbone.net.layers[-1]),copy.deepcopy(self.backbone.net.final),rearrange_layer)
            layer_weight, final_weight = self.distiall_head_weight(self.branch3_weight_path)
            self.branch3[0].load_state_dict(layer_weight)
            self.branch3[1].load_state_dict(final_weight)

            # self.branch6  = nn.Sequential(copy.deepcopy(self.backbone.net.layers[-1]), copy.deepcopy(self.backbone.net.final), rearrange_layer)
            # layer_weight, final_weight = self.distiall_head_weight("checkpoints/WeathBench32x64/LgNet_Head/ts_2_finetune-2D706N_per_24_step")
            # self.branch1[0].load_state_dict(layer_weight)
            # self.branch1[1].load_state_dict(final_weight)

        @staticmethod
        def distiall_head_weight(ckpt):
            ckpt = torch.load(ckpt)
            model_state_dict = ckpt['model']
            if "loragrashcastdglsym" in model_state_dict:
                model_state_dict = model_state_dict["loragrashcastdglsym"]
            if "lgnet" in model_state_dict:
                model_state_dict = model_state_dict["lgnet"]


            layer_dict = {}
            for key,val in model_state_dict.items():
                if "layers.2" in key:
                    pass
                else:continue
                key = key.replace("module.","").replace("_orig_mod.","")
                key = key.replace("backbone.net.layers.2.","")
                layer_dict[key] = val

            final_dict = {}
            for key,val in model_state_dict.items():
                if "final" in key:
                    pass
                else:continue
                key = key.replace("module.","").replace("_orig_mod.","")
                key = key.replace("backbone.net.final.","")
                final_dict[key] = val
            return layer_dict , final_dict

        def decode(self, x, branch_flag):

            if branch_flag == 6:
                return self.backbone.decode(x)
            elif branch_flag == 1:
                return self.branch1(x)
            elif branch_flag == 3:
                return self.branch3(x)
            elif branch_flag == 24:
                return self.branch24(x)
            else:
                raise NotImplementedError

        def forward(self, x, branch_flag):
            assert isinstance(branch_flag, int)
            return self.decode(self.backbone.encode(x), branch_flag)

    class LgNet_MultiHead_F(LgNet_MultiHead):
        '''
            use it as wrapper model.
            base branch is 6;
        '''
        branch1_weight_path = "checkpoints/WeathBench32x64/LgNet_Head/ts_3_finetune-2D706N_per_1_step/03_11_20_50_45793-seed_73001/pretrain_latest.pt"
        branch24_weight_path = "checkpoints/WeathBench32x64/LgNet_Head/ts_3_finetune-2D706N_per_24_step/03_11_04_21_49028-seed_73001/pretrain_latest.pt"
        branch3_weight_path = "checkpoints/WeathBench32x64/LgNet_Head/ts_3_finetune-2D706N_per_3_step/03_11_19_51_49155-seed_73001/pretrain_latest.pt"

       

    class LgNet_Head(CK_LgNet):
        '''
            use it as wrapper model.
        '''
        def set_epoch(self,epoch=None,epoch_total=None,eval_mode=False):
            for p in self.backbone.parameters():p.requires_grad=False
            for p in self.backbone.net.layers[-1].parameters():p.requires_grad=True
            for p in self.backbone.net.final.parameters():p.requires_grad=True

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