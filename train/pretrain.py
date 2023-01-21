
import optuna
import os, sys,time,json,copy
sys.path.append(os.getcwd())
from gpu_use_setting import *
idx=0
sys.path = [p for p in sys.path if 'lustre' not in p]
experiment_hub_path = "./experiment_hub.json"
force_big  = True
save_intervel=100
import hashlib
import wandb
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from torch.nn.parallel import DistributedDataParallel
import timm.optim
from timm.scheduler import create_scheduler
#import hfai
#hfai.set_watchdog_time(21600)
#import hfai.nccl.distributed as dist
import torch.distributed as dist
# from hfai.nn.parallel import DistributedDataParallel
#from ffrecord.torch import DataLoader
#import hfai.nn as hfnn
#from hfai.datasets import ERA5
from model.afnonet import AFNONet
from model.FEDformer import FEDformer
from model.FEDformer1D import FEDformer1D
from JCmodels.fourcastnet import AFNONet as AFNONetJC
from model.patch_model import *
from model.time_embeding_model import *
from model.physics_model import *
from model.othermodels import *
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import single_step_evaluate
import pandas as pd

from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.loggingsystem import LoggingSystem

from model.GradientModifier import *
from cephdataset import *
# dataset_type = ERA5CephDataset
# dataset_type  = SpeedTestDataset
Datafetcher = DataSimfetcher
def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.
import random
import traceback
from mltool.visualization import *
import time

lr_for_mode={'pretrain':5e-4,'finetune':1e-4,'fourcast':1e-4}
ep_for_mode={'pretrain':80,'finetune':50,'fourcast':50}
bs_for_mode={'pretrain':4,'finetune':3,'fourcast':3}
as_for_mode={'pretrain':8,'finetune':16,'fourcast':16}
ts_for_mode={'pretrain':2,'finetune':3,'fourcast':36}

train_set={
    'large': ((720, 1440), 8, 20, 20, ERA5CephDataset,{}),
    'small': ( (32,   64), 8, 20, 20, ERA5CephSmallDataset,{}),
    'test_large': ((720, 1440), 8, 20, 20, lambda **kargs:SpeedTestDataset(720,1440,**kargs),{}),
    'test_small': ( (32,   64), 8, 20, 20, lambda **kargs:SpeedTestDataset( 32,  64,**kargs),{}),
    'physics_small': ( (32,   64), 2, 12, 12, ERA5CephSmallDataset,{'dataset_flag':'physics'}),
    'physics': ((720, 1440), 8, 12, 12, ERA5CephDataset,{'dataset_flag':'physics'}),
    '4796normal': ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normal','use_scalar_advection':False}),
    '4796Rad' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'reduce','use_scalar_advection':True}),
    '4796ad' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normal','use_scalar_advection':True}),
    '4796Field' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'only_Field','use_scalar_advection':False}),
    '4796FieldNormlized' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normalized_data','use_scalar_advection':False}),
    '2D70N':((32,64), (2,2), 70, 70, WeathBench71,{'dataset_flag':'2D70N'}),
    '2D7066N':((32,64), (2,2), 70, 70, WeathBench7066,{'dataset_flag':'2D70N'}),
    '2D70NH5':((32,64), (2,2), 70, 70, WeathBench71_H5,{'dataset_flag':'2D70N'}),
    '2D706N':((32,64), (2,2), 70, 70, WeathBench706,{'dataset_flag':'2D70N'}),
    '2D716N':((32,64), (2,2), 71, 71, WeathBench716,{'dataset_flag':'2D71N'}),
    '3D70N':((14,32,64), (2,2,2), 5, 5, WeathBench71,{'dataset_flag':'3D70N'}),
}
#img_size, patch_size, x_c, y_c, dataset_type = train_set[args.train_set]


half_model = False




#########################################
########### metric computing ############
#########################################
def generate_latweight(w, device):
    # steph = 180.0 / h
    # latitude = np.arange(-90, 90, steph).astype(np.int)
    tw =  32 if w < 32 else w
    latitude = torch.linspace(-np.pi/2,np.pi/2,tw).to(device)
    if w<32:
        offset  = ( 32 - w )//2
        latitude=latitude[offset:-offset]
    cos_lat   = torch.cos(latitude)
    latweight = cos_lat/cos_lat.mean()
    latweight = latweight.reshape(1, w, 1,1)
    return latweight

def compute_accu(ltmsv_pred, ltmsv_true):
    if len(ltmsv_pred.shape)==5:ltmsv_pred = ltmsv_pred.flatten(1,2)
    if len(ltmsv_true.shape)==5:ltmsv_true = ltmsv_true.flatten(1,2)
    ltmsv_pred = ltmsv_pred.permute(0,2,3,1)
    ltmsv_true = ltmsv_true.permute(0,2,3,1)
    # ltmsv_pred --> (B, w, h, property)
    # ltmsv_true --> (B, w, h, property)
    latweight = generate_latweight(ltmsv_pred.shape[1],ltmsv_pred.device)
    # history_record <-- (B, w,h, property)
    fenzi = (latweight*ltmsv_pred*ltmsv_true).sum(dim=(1, 2))
    fenmu = torch.sqrt((latweight*ltmsv_pred**2).sum(dim=(1,2)) *
                       (latweight*ltmsv_true**2).sum(dim=(1, 2))
                       )
    return torch.clamp(fenzi/(fenmu+1e-10),0,10)

def compute_rmse(pred, true, return_map_also=False):
    if len(pred.shape)==5:pred = pred.flatten(1,2)
    if len(true.shape)==5:true = true.flatten(1,2)
    pred = pred.permute(0,2,3,1)
    true = true.permute(0,2,3,1)
    latweight = generate_latweight(pred.shape[1],pred.device)
    out = torch.sqrt(torch.clamp((latweight*(pred - true)**2).mean(dim=(1,2)),0,1000))
    if return_map_also:
        out = [out, torch.clamp((latweight*(pred - true)**2).sum(dim=(0)),0,1000)]
    return out


#########################################
########## normal forward step ##########
#########################################

def make_data_regular(batch,half_model=False):
    # the input can be
    # [
    #   timestamp_1:[Field,Field_Dt,(physics_part) ],
    #   timestamp_2:[Field,Field_Dt,(physics_part) ],
    #     ...............................................
    #   timestamp_n:[Field,Field_Dt,(physics_part) ]
    # ]
    # or
    # [
    #   timestamp_1:Field,
    #   timestamp_2:Field,
    #     ...............................................
    #   timestamp_n:Field
    # ]
    if not isinstance(batch,(list,tuple)):
        
        batch = batch.half() if half_model else batch.float()
        if len(batch.shape)==4:
            channel_last = batch.shape[1] in [32,720] # (B, P, W, H )
            if channel_last:batch = batch.permute(0,3,1,2)
        return batch
    else:
        return [make_data_regular(x,half_model=half_model) for x in batch]

def once_forward_with_timestamp(model,i,start,end,dataset,time_step_1_mode):
    if not isinstance(end[0],(list,tuple)):end = [end]
    start_timestamp= torch.stack([t[1] for t in start],1) #[B,T,4]
    end_timestamp = torch.stack([t[1] for t in end],1) #[B,T,4]    
    #print([(s[0].shape,s[1].shape) for s in start])
    # start is data list [ [[B,P,h,w],[B,4]] , [[B,P,h,w],[B,4]], [[B,P,h,w],[B,4]], ...]
    normlized_Field_list = dataset.do_normlize_data([[t[0] for t in start]])[0]  #always use normlized input
    normlized_Field    = torch.stack(normlized_Field_list,2) #(B,P,T,w,h)

    target_list = dataset.do_normlize_data([[t[0] for t in end]])[0]  #always use normlized input
    target      = torch.stack(target_list,2) #(B,P,T,w,h)
    
    if 'FED' in model.__class__.__name__:
        out  = model(normlized_Field, start_timestamp, end_timestamp)
    else:
        out = model(normlized_Field)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    if model.pred_len==1:
        out=out.squeeze(2)
        target=target.squeeze(2)
    ltmv_pred  = dataset.inv_normlize_data([out])[0]
    end_timestamp= end_timestamp.squeeze(1)
    start = start[1:]+[[ltmv_pred,end_timestamp]]
    
    # return:
    #  ltmv_pred [not normlized pred Field]
    #   target  [normlized target Field]
    #   extra_loss
    #   extra_info_from_model_list
    #   start [not normlized Field list]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_with_timeconstant(model,i,start,end,dataset,time_step_1_mode):
    assert len(start)==1
    # start = [[ (B,68,32,64), (B,1,32,64), (B,1,32,64)], [...], [...]]
    # end  = [ (B,68,32,64), (B,1,32,64), (B,1,32,64)]
    normlized_Field = torch.cat(start[0],1) #(B,P+2,w,h)
    target      = end[0]
    
    out = model(normlized_Field)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred  = dataset.inv_normlize_data([out])[0]

    start = start[1:]+[[ltmv_pred]+end[1:]]

    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_normal(model,i,start,end,dataset,time_step_1_mode):
    Field = Advection = None

    if isinstance(start[0],(list,tuple)):# now is [Field, Field_Dt, physics_part]
        Field  = start[-1][0] # the now Field is the newest timestamp
        Advection= start[-1][-1]
        normlized_Field_list = dataset.do_normlize_data(start)
        normlized_Field_list = [p[0] for p in normlized_Field_list]
        normlized_Field=normlized_Field_list[0] if len(normlized_Field_list)==1 else torch.stack(normlized_Field_list,1)
        #(B,P,y,x) for no history case (B,P,H,y,x) for history version
        target = dataset.do_normlize_data([end[0]])[0] # in standand unit
    elif time_step_1_mode:
        normlized_Field, target = dataset.do_normlize_data([[start,target]])[0]
    else:
        Field  = start[-1]
        if hasattr(model,'calculate_Advection'):Advection = model.calculate_Advection(Field)
        if hasattr(model,'module') and hasattr(model.module,'calculate_Advection'):Advection = model.module.calculate_Advection(Field)
        normlized_Field_list = dataset.do_normlize_data([start])[0]  #always use normlized input
        normlized_Field      = normlized_Field_list[0] if len(normlized_Field_list)==1 else torch.stack(normlized_Field_list,2)
        target               = dataset.do_normlize_data([end])[0] #always use normlized target

    if model.training and model.input_noise_std and i==1:
        normlized_Field += torch.randn_like(normlized_Field)*model.input_noise_std

    train_channel_from_this_stamp = None
    if hasattr(model,"train_channel_from_this_stamp"): train_channel_from_this_stamp = model.train_channel_from_this_stamp
    if hasattr(model,"module") and hasattr(model.module,"train_channel_from_this_stamp"): train_channel_from_this_stamp = model.module.train_channel_from_this_stamp
    train_channel_from_next_stamp = None
    if hasattr(model,"train_channel_from_next_stamp"): train_channel_from_next_stamp = model.train_channel_from_next_stamp
    if hasattr(model,"module") and hasattr(model.module,"train_channel_from_next_stamp"): train_channel_from_next_stamp = model.module.train_channel_from_next_stamp
    pred_channel_for_next_stamp = None
    if hasattr(model,"pred_channel_for_next_stamp"): pred_channel_for_next_stamp = model.pred_channel_for_next_stamp
    if hasattr(model,"module") and hasattr(model.module,"pred_channel_for_next_stamp"): pred_channel_for_next_stamp = model.module.pred_channel_for_next_stamp
    fourcast_for_2D70N = None
    if hasattr(model,"fourcast_for_2D70N"): fourcast_for_2D70N = model.fourcast_for_2D70N
    
    if train_channel_from_this_stamp:
        assert len(normlized_Field.shape)==4
        normlized_Field = normlized_Field[:,train_channel_from_this_stamp]

    if train_channel_from_next_stamp:
        assert len(normlized_Field.shape)==4
        normlized_Field = torch.cat([normlized_Field,target[:,train_channel_from_next_stamp]],1)

    
    #print(normlized_Field.shape,torch.std_mean(normlized_Field))
    out   = model(normlized_Field)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    if Advection is not None:
        normlized_Deltat_F  = out
        _, Deltat_F = dataset.inv_normlize_data([[0,normlized_Deltat_F]])[0]
        reduce_Field_coef = torch.Tensor(dataset.reduce_Field_coef).to(normlized_Deltat_F.device)
        if hasattr(model,'reduce_Field_coef'):reduce_Field_coef+=model.reduce_Field_coef
        ltmv_pred   = Field + Deltat_F - Advection*reduce_Field_coef
    else:
        ltmv_pred = dataset.inv_normlize_data([out])[0]

    if isinstance(start[0],(list,tuple)):
        start = start[1:]+[[ltmv_pred, 0 , end[-1]]]
    elif pred_channel_for_next_stamp:
        next_tensor = target.clone().type(ltmv_pred.dtype)
        next_tensor[:,pred_channel_for_next_stamp] = ltmv_pred
        start     = start[1:] + [next_tensor]
    elif fourcast_for_2D70N:
        next_tensor = target.clone().type(ltmv_pred.dtype)
        picked_property = list(range(0,14*4-1)) + list(range(14*4,14*5-1))
        next_tensor[:,picked_property] = ltmv_pred[:,picked_property]
        start     = start[1:] + [next_tensor]
    else:
        start     = start[1:] + [ltmv_pred]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))

    
    if pred_channel_for_next_stamp:
        target = target[:,pred_channel_for_next_stamp]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_patch(model,i,start,end,dataset,time_step_1_mode):
    time_stamp = None
    pos = None
    assert len(start)==1

    start_tensor = start
    if isinstance(start[-1],list):
        assert len(start[-1])<=3 # only allow tensor + time_stamp + pos
        tensor, time_stamp, pos = start[-1]
        start_tensor = [tensor]
    
 
    Field  = start_tensor[-1]
    normlized_Field_list = dataset.do_normlize_data([start_tensor])[0]  #always use normlized input
    normlized_Field    = normlized_Field_list[0] if len(normlized_Field_list)==1 else torch.stack(normlized_Field_list,2)
    

    if time_stamp is not None or pos is not None:
        target = dataset.do_normlize_data([end[0]])[0] #always use normlized target
    else:
        target = dataset.do_normlize_data([end])[0] #always use normlized target

    if model.training and model.input_noise_std and i==1:
        normlized_Field += torch.randn_like(normlized_Field)*model.input_noise_std


    if (time_stamp is not None) or (pos is not None) :
        out   = model(normlized_Field,time_stamp,pos)
    else:
        out   = model(normlized_Field)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = dataset.inv_normlize_data([out])[0]
    
    if isinstance(end,(list,tuple)):
        start = start[1:] + [[ltmv_pred, end[1] ,end[2]]]
    else:
        start     = start[1:] + [ltmv_pred]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))
    get_center_index_depend_on = model.module.get_center_index_depend_on if hasattr(model,'module') else model.get_center_index_depend_on
    if len(ltmv_pred.shape)>2: #(B,P,Z,H,W)
        if ltmv_pred.shape!=target.shape: # (B,P,W,H) -> (B,P,W-4,H) mode
            if len(target.shape) == 5:
                img_shape = target.shape[-3:]
                sld_shape = ltmv_pred.shape[-3:]
                z_idx, h_idx, l_idx = get_center_index_depend_on(sld_shape, img_shape)[0]
                target = target[...,z_idx, h_idx, l_idx]
            elif len(target.shape) == 4:
                img_shape = target.shape[-2:]
                sld_shape = ltmv_pred.shape[-2:]
                h_idx, l_idx = get_center_index_depend_on(sld_shape, img_shape)[0]
                target = target[...,h_idx, l_idx]
            else:
                raise NotImplementedError
        else:
            # (B,P,W,H) -> (B,P,W,H) mode
            pass
    else: #(B, P)
        #if ltmv_pred.shape[-1] == 1: # (B,P,5,5) -> (B,P) mode
        if len(target.shape) == 4:
            B,P,W,H=target.shape
            target = target[...,W//2,H//2]
        elif len(target.shape) == 5:
            B,P,Z,W,H=target.shape
            target = target[...,Z//2,W//2,H//2]  
        else:
            raise NotImplementedError
#         else:
#             # (B,P,5,5) -> (B,P) mode
#             target = target 
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_patch_N2M(model,i,start,end,dataset,time_step_1_mode):
    time_stamp = None
    pos = None
    assert len(start)==model.history_length
    assert len(end)  ==model.pred_len
    assert len(start[0])==3
    assert len(end[0])==3
    # start 
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    #         ..........
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]

    # end 
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    #         ..........
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    
    assert isinstance(start[-1],list)
    
    tensor, start_time_stamp, start_pos = [torch.stack([s[i] for s in start],1) for i in range(len(start[-1]))] 
    target,   end_time_stamp,   end_pos =  [torch.stack([s[i] for s in   end],1) for i in range(len(  end[-1]))] 
    
    out   = model(tensor, start_pos,start_time_stamp,end_time_stamp) #(B,T,P,W,H) (B,T,4) (B,T,2,W,H)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = dataset.inv_normlize_data([out])[0] #(B,T,P,W,H)
    
    start = start[len(end):] + [[tensor, time_stamp, pos] for tensor,(_,time_stamp,pos) in zip(ltmv_pred.permute(1,0,2,3,4), end)]
    
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_deltaMode(model,i,start,end,dataset,time_step_1_mode):
    assert len(start) == 1
    base1, delta1 = start[0]
    base2, delta2 = end
    out   = model(delta1)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    ltmv_pred = out
    target   = delta2
    dataset.delta_mean_tensor = dataset.delta_mean_tensor.to(base1.device)
    dataset.delta_std_tensor  = dataset.delta_std_tensor.to(base1.device)
    start   = start[1:] + [[base1 + (delta1*dataset.delta_std_tensor + dataset.delta_mean_tensor) ,ltmv_pred]]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start



def once_forward_self_relation(model,i,start,end,dataset,time_step_1_mode=None):
    assert len(start) == 1
    input_feature  = start[0]
    output_feature = end
    out   = model(input_feature)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out,(list,tuple)):
        extra_loss                 = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    ltmv_pred = out
    target    = output_feature
    start   = start[1:] + [None]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward(model,i,start,end,dataset,time_step_1_mode):
    if 'Patch' in dataset.__class__.__name__:
        if model.pred_len > 1:
            return once_forward_patch_N2M(model,i,start,end,dataset,time_step_1_mode)
        else:
            return once_forward_patch(model,i,start,end,dataset,time_step_1_mode)
    elif 'SolarLunaMask' in dataset.__class__.__name__:
        return once_forward_with_timeconstant(model,i,start,end,dataset,time_step_1_mode)
    elif hasattr(dataset,'use_time_stamp') and dataset.use_time_stamp:
        return once_forward_with_timestamp(model,i,start,end,dataset,time_step_1_mode)
    elif 'Delta' in dataset.__class__.__name__:
        return once_forward_deltaMode(model,i,start,end,dataset,time_step_1_mode)
    elif dataset.__class__.__name__ == "WeathBench7066Self":
        return once_forward_self_relation(model,i,start,end,dataset,time_step_1_mode)
    else:   
       return  once_forward_normal(model,i,start,end,dataset,time_step_1_mode)

def run_one_iter(model, batch, criterion, status, gpu, dataset):
    iter_info_pool={}
    loss = 0
    diff = 0
    random_run_step = np.random.randint(1,len(batch)) if len(batch)>1 else 0
    time_step_1_mode=False
    if len(batch) == 1 and isinstance(batch[0],(list,tuple)) and len(batch[0])>1:
        batch = batch[0] # (Field, FieldDt)
        time_step_1_mode=True
    if model.history_length > len(batch):
        print(f"you want to use history={model.history_length}")
        print(f"but your input batch(timesteps) only has len(batch)={len(batch)}")
        raise
    pred_step = 0
    start = batch[0:model.history_length] # start must be a list
    
    for i in range(model.history_length,len(batch), model.pred_len):# i now is the target index
        end = batch[i:i+model.pred_len]
        end = end[0] if len(end) == 1 else end
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,end,dataset,time_step_1_mode)

        if extra_loss !=0:
            iter_info_pool[f'{status}_extra_loss_gpu{gpu}_timestep{i}'] = extra_loss.item()
        for extra_info_from_model in extra_info_from_model_list:
            for name, value in extra_info_from_model.items():
                iter_info_pool[f'valid_on_{status}_{name}_timestep{i}'] = value
        
        ltmv_pred = dataset.do_normlize_data([ltmv_pred])[0]

        if 'Delta' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)
            with torch.no_grad():
                normlized_field_predict = dataset.combine_base_delta(start[-1][0], start[-1][1]) 
                normlized_field_real    = dataset.combine_base_delta(      end[0],       end[1])  
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
            
        elif 'deseasonal' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)
            with torch.no_grad():
                normlized_field_predict = dataset.addseasonal(start[-1][0], start[-1][1])
                normlized_field_real    = dataset.addseasonal(end[0], end[1])
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
        elif '68pixelnorm' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)
            with torch.no_grad():
                normlized_field_predict = dataset.recovery(start[-1])
                normlized_field_real    = dataset.recovery(end)
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
        else:
            normlized_field_predict = ltmv_pred
            normlized_field_real = target
            abs_loss = criterion[pred_step](ltmv_pred,target) if isinstance(criterion,(dict,list)) else criterion(ltmv_pred,target)
            loss += abs_loss + extra_loss
        
        diff += abs_loss
        pred_step+=1
        

        iter_info_pool[f'{status}_abs_loss_gpu{gpu}_timestep{i}'] =  abs_loss.item()
        if status != "train":
            iter_info_pool[f'{status}_accu_gpu{gpu}_timestep{i}']     =  compute_accu(normlized_field_predict,normlized_field_real).mean().item()
            iter_info_pool[f'{status}_rmse_gpu{gpu}_timestep{i}']     =  compute_rmse(normlized_field_predict,normlized_field_real).mean().item()
        if model.random_time_step_train and i >= random_run_step:
            break
    # loss = loss/(len(batch) - 1)
    # diff = diff/(len(batch) - 1)
    loss = loss/pred_step
    diff = diff/pred_step
    return loss, diff, iter_info_pool, ltmv_pred, target


class RandomSelectPatchFetcher:
    def __init__(self,data_loader,device):
        dataset = data_loader.dataset
        assert dataset.use_offline_data  
        self.data  = dataset.dataset_tensor #(B,70,32,64)
        self.batch_size = data_loader.batch_size 
        self.patch_range= dataset.patch_range
        self.img_shape  = dataset.img_shape
        self.around_index = dataset.around_index
        self.center_index = dataset.center_index
        self.length = len(dataset)
        self.time_step = dataset.time_step
        self.device = device
        self.use_time_stamp = dataset.use_time_stamp
        self.use_position_idx= dataset.use_position_idx
        self.timestamp = dataset.timestamp
    def next(self):
        if len(self.img_shape)==2:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2,size=(self.batch_size,)) 
            center_w = np.random.randint(self.img_shape[-1],size=(self.batch_size,))
            location = self.around_index[center_h, center_w] #(B,2,5,5) 
            patch_idx_h = location[:,0]#(B,5,5)
            patch_idx_w = location[:,1]#(B,5,5)
            #location  = self.center_index[:, center_h, center_w].transpose(1,0)#(B,2)
            batch_idx = np.random.randint(self.length,size=(self.batch_size,)).reshape(self.batch_size,1,1) #(B,1,1)
            data = [[self.data[batch_idx+i,:,patch_idx_h,patch_idx_w].permute(0,3,1,2).to(self.device)] for i in range(self.time_step)]
        elif len(self.img_shape)==3:
            center_z    = np.random.randint(self.img_shape[-3] - (self.patch_range[-3]//2)*2,size=(self.batch_size,)) 
            center_h    = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2,size=(self.batch_size,))  
            center_w    = np.random.randint(self.img_shape[-1],size=(self.batch_size,)) 
            location    = self.around_index[center_z, center_h, center_w] #(B,2,5,5,5) 
            patch_idx_z = location[:,0]#(B,5,5,5)
            patch_idx_h = location[:,1]#(B,5,5,5)
            patch_idx_w = location[:,2]#(B,5,5,5)
            #location = self.center_index[:,center_z, center_h, center_w].transpose(1,0)#(B,3)
            batch_idx = np.random.randint(self.length,size=(self.batch_size,)).reshape(self.batch_size,1,1,1) #(B,1,1,1)
            data = [[self.data[batch_idx+i,:,patch_idx_z,patch_idx_h,patch_idx_w].permute(0,4,1,2,3).to(self.device)] for i in range(self.time_step)]
        else:
            raise NotImplementedError
        out = data 
        if self.use_time_stamp:
            out = [out[i]+[torch.Tensor(self.timestamp[batch_idx.flatten()+i]).to(self.device)] for i in range(self.time_step)]
        if self.use_position_idx:
            out = [out[i]+[torch.Tensor(location).to(self.device)] for i in range(self.time_step)]
        if len(out[0])==1:
            out = [t[0] for t in out]
        return out 

def run_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    if optimizer.grad_modifier and optimizer.grad_modifier.__class__.__name__=='NGmod_RotationDeltaEThreeTwo':
        assert data_loader.dataset.time_step==3
        return run_one_epoch_three2two(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status)
    else:
        return run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status)

def run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    
    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    accumulation_steps = model.accumulation_steps # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost  = []
    train_cost = []
    rest_cost  = []
    now = time.time()

    Fethcher   = RandomSelectPatchFetcher if( status =='train' and \
                                              data_loader.dataset.use_offline_data and \
                                              data_loader.dataset.split=='train' and \
                                              'Patch' in data_loader.dataset.__class__.__name__) else Datafetcher
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    #raise
    batches    = len(data_loader)

    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    #intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    path_loss = path_length = rotation_loss = None
    didunscale = False
    grad_modifier = optimizer.grad_modifier
    skip = False
    count_update = 0
    
    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        
        run_gmod = False
        if grad_modifier is not None:
            control = step if grad_modifier.update_mode==2 else count_update
            run_gmod = (control%grad_modifier.ngmod_freq==0)

        batch = prefetcher.next()

        # In this version(2022-12-22) we will split normal and ngmod processing
        # we will do normal train with normal batchsize and learning rate multitimes
        # then do ngmod train 
        # Notice, one step = once backward not once forward
        
        #[print(t[0].shape) for t in batch]
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now);now = time.time()
        if status == 'train':
            if hasattr(model,'set_step'):model.set_step(step=step,epoch=epoch)
            if hasattr(model,'module') and hasattr(model.module,'set_step'):model.module.set_step(step=step,epoch=epoch)
            # if model.train_mode =='pretrain':
            #     time_truncate = max(min(epoch//3,data_loader.dataset.time_step),2)
            #     batch=batch[:model.history_length -1 + time_truncate]

            # the normal initial method will cause numerial explore by using timestep > 4 senenrio.
            
            if grad_modifier is not None and run_gmod and (grad_modifier.update_mode==2):
                chunk = grad_modifier.split_batch_chunk
                ng_accu_times = max(data_loader.batch_size//chunk,1)

                batch_data_full = batch[0]
                
                ## nodal loss
                #### to avoid overcount,
                ## use model.module rather than model in Distribution mode is fine.
                # It works, although I think it is not safe. 
                # use model in distribution mode will go wrong, altough it can work in old code version.
                # I suppose it is related to the graph optimization processing in pytorch.
                for chunk_id in range(ng_accu_times):
                    if isinstance(batch_data_full,list):
                        batch_data = torch.cat([ttt[chunk_id*chunk:(chunk_id+1)*chunk].flatten(1,-1) for ttt in batch_data_full],1)
                    else:
                        batch_data = batch_data_full[chunk_id*chunk:(chunk_id+1)*chunk]
                    ngloss=None
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        if grad_modifier.lambda1!=0:
                            if ngloss is None:ngloss=0
                            Nodeloss1 = grad_modifier.getL1loss(model.module if hasattr(model,'module') else model, batch_data,coef=grad_modifier.coef)/ng_accu_times
                            ngloss  += grad_modifier.lambda1 * Nodeloss1
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            if ngloss is None:ngloss=0
                            Nodeloss2 = grad_modifier.getL2loss(model.module if hasattr(model,'module') else model, batch_data,coef=grad_modifier.coef)/ng_accu_times
                            ngloss += grad_modifier.lambda2 * Nodeloss2
                            Nodeloss2=Nodeloss2.item()
                    if ngloss is not None:
                        loss_scaler.scale(ngloss).backward()    
                        # if model.use_amp:
                        #     loss_scaler.scale(ngloss).backward()    
                        # else:
                        #     ngloss.backward()

                    # for idx,(name,p) in enumerate(model.named_parameters()):
                    #     print(f"{chunk_id}:{name}:{p.device}:{p.norm()}:{p.grad.norm() if p.grad is not None else None}")
                    #     if idx>10:break
                    # print("===========================")
                #raise


            with torch.cuda.amp.autocast(enabled=model.use_amp):
                loss, abs_loss, iter_info_pool,ltmv_pred,target  =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
                ## nodal loss
                if (grad_modifier is not None) and (run_gmod) and (grad_modifier.update_mode==2):
                    if grad_modifier.lambda1!=0:
                        Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                        loss += grad_modifier.lambda1 * Nodeloss1
                        Nodeloss1=Nodeloss1.item()
                    if grad_modifier.lambda2!=0:
                        Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                        if Nodeloss2>0:
                            loss += grad_modifier.lambda2 * Nodeloss2
                        Nodeloss2=Nodeloss2.item()
            
            loss, nan_count, skip = nan_diagnose_weight(model,loss,nan_count,logsys)
            if skip:continue
            
            loss /= accumulation_steps
            loss_scaler.scale(loss).backward()    
            # else:
            #     loss.backward()

            #select_para= list(range(5)) + [-5,-4,-3,-2,-1] 
            
            #pnormlist = [[name,p.grad.norm()] for name, p in model.named_parameters() if p.grad is not None]
            #for i in select_para:
            #    name,norm = pnormlist[i]
            #    print(f'before:gpu:{device} - {name} - {norm}')
            
            path_loss = path_length = None
            if grad_modifier and grad_modifier.path_length_regularize and step%grad_modifier.path_length_regularize==0:
                mean_path_length = model.mean_path_length.to(device)
                
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    path_loss, mean_path_length, path_lengths = grad_modifier.getPathLengthloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], mean_path_length, path_length_mode=grad_modifier.path_length_mode )
                
                if path_loss > grad_modifier.loss_wall:
                    the_loss = path_loss*grad_modifier.gd_alpha
                    loss_scaler.scale(the_loss).backward()    
                # if grad_modifier.use_amp:
                #     loss_scaler.scale(the_loss).backward()    
                # else:
                #     the_loss.backward()

                #pnormlist = [[name,p.grad] for name, p in model.named_parameters() if p.grad is not None]
                #name,norm = pnormlist[0]
                #print(f'before:gpu:{device} - {name} - {norm}')
                #for i in select_para:
                #    name,norm = pnormlist[i]
                #    print(f'before:gpu:{device} - {name} - {norm}')

                if hasattr(model,'module'):
                    mean_path_length = mean_path_length/dist.get_world_size()
                    # dist.barrier()# <--- its doesn't matter
                    dist.all_reduce(mean_path_length)
                    
                #pnormlist = [[name,p.grad] for name, p in model.named_parameters() if p.grad is not None]
                #name,norm = pnormlist[0]
                #print(f'before:gpu:{device} - {name} - {norm}')
                #for i in select_para:
                #    name,norm = pnormlist[i]
                #    print(f'after:gpu:{device} - {name} - {norm}')
                   
                model.mean_path_length = mean_path_length.detach().cpu()
                path_loss = path_loss.item()
                path_lengths=path_lengths.mean().item()

                
            rotation_loss = None
            if grad_modifier and grad_modifier.rotation_regularize and step%grad_modifier.rotation_regularize==0:
                # amp will destroy the train
                #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
                #rotation_loss.backward()
                if grad_modifier.only_eval:
                    with torch.no_grad(): 
                        with torch.cuda.amp.autocast(enabled=model.use_amp):
                            rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)                     
                else:
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)                    
                    if grad_modifier.alpha_stratagy == 'softwarmup50.90':
                        gd_alpha = grad_modifier.gd_alpha*min(max((np.exp((epoch-50)/40)-1)/(np.exp(1)-1),0),1)
                    elif grad_modifier.alpha_stratagy == 'normal':
                        gd_alpha=grad_modifier.gd_alpha
                    else:
                        raise NotImplementedError

                    if (rotation_loss > grad_modifier.loss_wall) and gd_alpha>0: #default grad_modifier.loss_wall is 0
                        if grad_modifier.loss_target:
                            the_loss = abs(rotation_loss-grad_modifier.loss_target)*gd_alpha
                        else:
                            the_loss = rotation_loss*gd_alpha
                        loss_scaler.scale(the_loss).backward() 
                    # if grad_modifier.use_amp:
                    #     loss_scaler.scale(the_loss).backward()    
                    # else:
                    #     the_loss.backward()
                
                rotation_loss = rotation_loss.item()
                
            # In order to use multiGPU train, I have to use Loss update scenario, suprisely, it is not worse than split scenario
            # if optimizer.grad_modifier is not None:
            #     #assert not model.use_amp
            #     #assert accumulation_steps == 1 
            #     if model.use_amp and not didunscale:
            #         loss_scaler.unscale_(optimizer) # do unscaler here for right gradient modify like clip or norm
            #         didunscale = True
            #     assert len(batch)==2 # we now only allow one 
            #     assert isinstance(batch[0],torch.Tensor)
            #     with controlamp(model.use_amp)():
            #         optimizer.grad_modifier.backward(model, batch[0], batch[1], strict=False)
            #         Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
            

            #nan_count, skip = nan_diagnose_grad(model,nan_count,logsys)
            # if skip:
            #     optimizer.zero_grad()
            #     continue
            if hasattr(model,'module') and grad_modifier:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/dist.get_world_size()
                        dist.all_reduce(p.grad) #<--- pytorch DDP doesn't support high order gradient. This step need!

            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if (step+1) % accumulation_steps == 0:
                loss_scaler.step(optimizer)
                loss_scaler.update()   
                # if model.use_amp:                  
                #     loss_scaler.step(optimizer)
                #     loss_scaler.update()   
                # else:
                #     optimizer.step()
                count_update += 1
                optimizer.zero_grad()
                didunscale = False
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    loss, abs_loss, iter_info_pool,ltmv_pred,target =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
                    if optimizer.grad_modifier is not None:
                        if grad_modifier.lambda1!=0:
                            Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss2=Nodeloss2.item()
        if logsys.do_iter_log > 0:
            if logsys.do_iter_log ==  1:iter_info_pool={} # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}']       =  loss.item()
        else:
            iter_info_pool={}
        if Nodeloss1  > 0:iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}']  = Nodeloss1
        if Nodeloss2  > 0:iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}']  = Nodeloss2
        if path_loss is not None:iter_info_pool[f'{status}_path_loss_gpu{gpu}']  = path_loss
        if path_length is not None:iter_info_pool[f'{status}_path_length_gpu{gpu}']  = path_length
        if rotation_loss is not None:iter_info_pool[f'{status}_rotation_loss_gpu{gpu}']= rotation_loss
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            #print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost  = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")


    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    return loss_val


def run_one_epoch_three2two(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    
    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    accumulation_steps = model.accumulation_steps # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost  = []
    train_cost = []
    rest_cost  = []
    now = time.time()

    Fethcher   = RandomSelectPatchFetcher if( status =='train' and \
                                              data_loader.dataset.use_offline_data and \
                                              data_loader.dataset.split=='train' and \
                                              'Patch' in data_loader.dataset.__class__.__name__) else Datafetcher
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    #raise
    batches    = len(data_loader)

    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    #intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    path_loss = path_length = rotation_loss = None
    didunscale = False
    grad_modifier = optimizer.grad_modifier
    skip = False
    count_update = 0
    
    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        
        run_gmod = False
        if grad_modifier is not None:
            control = step if grad_modifier.update_mode==2 else count_update
            run_gmod = (control%grad_modifier.ngmod_freq==0)

        batch = prefetcher.next()

        # In this version(2022-12-22) we will split normal and ngmod processing
        # we will do normal train with normal batchsize and learning rate multitimes
        # then do ngmod train 
        # Notice, one step = once backward not once forward
        
        #[print(t[0].shape) for t in batch]
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now);now = time.time()
        assert len(batch) == 3
        if status == 'train':
            if hasattr(model,'set_step'):model.set_step(step=step,epoch=epoch)
            if hasattr(model,'module') and hasattr(model.module,'set_step'):model.module.set_step(step=step,epoch=epoch)
            
            # one batch is [(B,P,W,H),(B,P,W,H),(B,P,W,H)]
            # the the input should be 
            batch = [torch.cat([batch[0],batch[1]]),torch.cat([batch[1],batch[2]])]
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                loss, abs_loss, iter_info_pool,ltmv_pred,target  =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            
            loss, nan_count, skip = nan_diagnose_weight(model,loss,nan_count,logsys)
            if skip:continue
            
            loss /= accumulation_steps
            loss_scaler.scale(loss).backward()    

            # if model.use_amp:
            #     loss_scaler.scale(loss).backward()    
            # else:
            #     loss.backward()
  
            rotation_loss = None
            if grad_modifier and grad_modifier.rotation_regularize and step%grad_modifier.rotation_regularize==0:
                # amp will destroy the train
                #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
                #rotation_loss.backward()
                if grad_modifier.only_eval:
                    with torch.no_grad(): 
                        with torch.cuda.amp.autocast(enabled=model.use_amp):
                            rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, 
                                    batch[0], 
                                    ltmv_pred.detach(),
                                    target,
                                    rotation_regular_mode = grad_modifier.rotation_regular_mode)                     
                else:
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)                    
                    the_loss = rotation_loss*grad_modifier.gd_alpha
                    if grad_modifier.use_amp:
                        loss_scaler.scale(the_loss).backward()    
                    else:
                        the_loss.backward()
                        
                rotation_loss = rotation_loss.item()


            # In order to use multiGPU train, I have to use Loss update scenario, suprisely, it is not worse than split scenario
            # if optimizer.grad_modifier is not None:
            #     #assert not model.use_amp
            #     #assert accumulation_steps == 1 
            #     if model.use_amp and not didunscale:
            #         loss_scaler.unscale_(optimizer) # do unscaler here for right gradient modify like clip or norm
            #         didunscale = True
            #     assert len(batch)==2 # we now only allow one 
            #     assert isinstance(batch[0],torch.Tensor)
            #     with controlamp(model.use_amp)():
            #         optimizer.grad_modifier.backward(model, batch[0], batch[1], strict=False)
            #         Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
            

            #nan_count, skip = nan_diagnose_grad(model,nan_count,logsys)
            # if skip:
            #     optimizer.zero_grad()
            #     continue
            if hasattr(model,'module') and grad_modifier:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/dist.get_world_size()
                        dist.all_reduce(p.grad) #<--- pytorch DDP doesn't support high order gradient. This step need!

            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if (step+1) % accumulation_steps == 0:
                loss_scaler.step(optimizer)
                loss_scaler.update()   
                # if model.use_amp:                  
                #     loss_scaler.step(optimizer)
                #     loss_scaler.update()   
                # else:
                #     optimizer.step()
                count_update += 1
                optimizer.zero_grad()
                didunscale = False
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    loss, abs_loss, iter_info_pool,ltmv_pred,target =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
                    if optimizer.grad_modifier is not None:
                        if grad_modifier.lambda1!=0:
                            Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss2=Nodeloss2.item()
        if logsys.do_iter_log > 0:
            if logsys.do_iter_log ==  1:iter_info_pool={} # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}']       =  loss.item()
        else:
            iter_info_pool={}
        if Nodeloss1  > 0:iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}']  = Nodeloss1
        if Nodeloss2  > 0:iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}']  = Nodeloss2
        if path_loss is not None:iter_info_pool[f'{status}_path_loss_gpu{gpu}']  = path_loss
        if path_length is not None:iter_info_pool[f'{status}_path_length_gpu{gpu}']  = path_length
        if rotation_loss is not None:iter_info_pool[f'{status}_rotation_loss_gpu{gpu}']= rotation_loss
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            #print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost  = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")


    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    return loss_val
 


def nan_diagnose_weight(model,loss, nan_count,logsys):
    skip = False
    if torch.isnan(loss):
        # we will check whether weight has nan 
        bad_weight_name = []
        bad_check = False
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                bad_check    = True
                bad_weight_name.append(name)
        if bad_check:
            logsys.info(f"the value is nan in weight:{bad_weight_name}")
            raise optuna.TrialPruned()
        else:
            nan_count+=1
            if nan_count>10:
                logsys.info("too many nan happened")
                raise optuna.TrialPruned()
            logsys.info(f"detect nan, now at {nan_count}/10 warning level, pass....")   
            skip = True
    return loss, nan_count, skip

def nan_diagnose_grad(model,nan_count,logsys):
    skip = False
    # we will check whether weight has nan 
    bad_weight_name = []
    bad_check = False
    for name, p in model.named_parameters():
        if p.grad is None:continue
        if torch.isnan(p.grad).any():
            bad_check    = True
            bad_weight_name.append(name)
    if bad_check:
        logsys.info(f"the value is nan in weight.grad:{bad_weight_name}")
        nan_count+=1
        if nan_count>10:
            logsys.info("too many nan happened")
            raise
        logsys.info(f"detect nan, now at {nan_count}/10 warning level, pass....")   
        skip = True
    return nan_count, skip



#########################################
######### fourcast forward step #########
#########################################
def save_and_log_table(_list, logsys, name, column, row=None):
    table= pd.DataFrame(_list.transpose(1,0),index=column, columns=row)
    new_row = [[a]+b for a,b in zip(row,_list.tolist())]
    logsys.add_table(name, new_row , 0, ['fourcast']+column)
    logsys.info(f"===>{name}<===")
    logsys.info(table);
    table.to_csv(os.path.join(logsys.ckpt_root,name))

def get_tensor_value(tensor,snap_index,time = 0):
    regist_batch_id_list, regist_feature_id_list, regist_position = snap_index
    tensor = tensor[0] if isinstance(tensor,list) else tensor
    if isinstance(regist_position,dict):
        regist_position = regist_position[time]
    # regist_batch_id_list is a list for select batch id
    # regist_feature_id_list is a list for select property id
    # regist_position is a position, if is 2D, it should be (#select_points,) (#select_points,)
    if len(tensor.shape) == 5:tensor = tensor.flatten(1,2) 
    output_tensor = []
    for regist_batch_id in regist_batch_id_list:
        one_batch_tensor= []
        for regist_feature_id in regist_feature_id_list:
            if len(regist_position)==2:
                location_tensor= tensor[regist_batch_id][regist_feature_id][regist_position[0],regist_position[1]]
            elif len(regist_position)==3:
                location_tensor= tensor[regist_batch_id,regist_feature_id,regist_position[0],regist_position[1],regist_position[2]]
            else:
                raise NotImplementedError
            one_batch_tensor.append(location_tensor.detach().cpu())
        output_tensor.append(torch.stack(one_batch_tensor))
    return torch.stack(output_tensor)#(B,P,N)

def calculate_next_level_error_once(model, x_t_1, error):
    '''
    calculate m(x_t_1)(x_t_1 - \hat{x_t_1})
    '''
    assert len(x_t_1.shape) == 4
    grad       = functorch.jvp(model, (x_t_1,), (error,))[1]  #(B, Xdimension)
    return grad

def calculate_next_level_error(model, x_t_1, error_list):
    '''
    calculate m(x_t_1)(x_t_1 - \hat{x_t_1})
    '''
    assert len(x_t_1.shape) == 4
    grads = vmap(calculate_next_level_error_once, (None,None, 0),randomness='same')(model,x_t_1,error_list)#(N, B, Xdimension)
    return grads

def get_tensor_norm(tensor,dim):#<--use mse way
    #return (torch.sum(tensor**2,dim=dim)).sqrt()#(N,B)
    return (torch.mean(tensor**2,dim=dim))#(N,B)

def create_multi_epoch_inference(fourcastresult_path_list, logsys,test_dataset,collect_names=['500hPa_geopotential','850hPa_temperature']):
    origin_ckpt_path = logsys.ckpt_root
    row=[]
    for epoch, fourcastresult in enumerate(fourcastresult_path_list):
        assert isinstance(fourcastresult,str)
        prefix = os.path.split(fourcastresult)[-1]
        #logsys.ckpt_root = fourcastresult
        # then it is the fourcastresult path
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'fourcastresult.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val
                else:
                    if key == 'global_rmse_map':

                        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],tmp['global_rmse_map'])]
                    else:
                        fourcastresult[key] = val # overwrite
        property_names = test_dataset.vnames
        if hasattr(test_dataset,"pred_channel_for_next_stamp"):
            property_names = [property_names[t] for t in test_dataset.pred_channel_for_next_stamp]
            test_dataset.unit_list = test_dataset.unit_list[test_dataset.pred_channel_for_next_stamp]
        accu_list = [p['accu'].cpu() for p in fourcastresult.values() if 'accu' in p]
        if len(accu_list)==0:continue
        accu_list = torch.stack(accu_list).numpy()   
        
        total_num = len(accu_list)
        accu_list = accu_list.mean(0)# (fourcast_num,property_num)
        real_times = [(predict_time+1)*test_dataset.time_intervel*test_dataset.time_unit for predict_time in range(len(accu_list))]
        #accu_table = save_and_log_table(accu_list,logsys, prefix+'accu_table', property_names, real_times)    

        ## <============= RMSE ===============>
        rmse_list = torch.stack([p['rmse'].cpu() for p in fourcastresult.values() if 'rmse' in p]).mean(0)# (fourcast_num,property_num)
        #rmse_table= save_and_log_table(rmse_list,logsys, prefix+'rmse_table', property_names, real_times)       

    
        if not isinstance(test_dataset.unit_list,int):
            unit_list = torch.Tensor(test_dataset.unit_list).to(rmse_list.device)
            #print(unit_list)
            unit_num  = max(unit_list.shape)
            unit_num  = len(property_names)
            unit_list = unit_list.reshape(1,unit_num)
            property_num = len(property_names)
            if property_num > unit_num:
                assert property_num%unit_num == 0
                unit_list = torch.repeat_interleave(unit_list,int(property_num//unit_num),dim=1)
        else:
            logsys.info(f"unit list is int, ")
            unit_list= test_dataset.unit_list
        
        rmse_unit_list= (rmse_list*unit_list)
        row += [[time_stamp,epoch]+value_list for time_stamp, value_list in zip(real_times,rmse_unit_list.tolist())]

    #logsys.add_table(prefix+'_rmse_unit_list', row , 0, ['fourcast']+['epoch'] + property_names)
    logsys.add_table('multi_epoch_fourcast_rmse_unit_list', row , 0, ['fourcast']+['epoch'] + property_names)
    

def get_error_propagation(last_pred, last_target, now_target, now_pred, virtual_function,approx_epsilon_lists, tangent_position='right'):
    the_abs_error_measure = None
    the_est_error_measure = None
    epsilon_alevel_v_real = None
    epsilon_blevel_v_real = None
    epsilon_Jacobian_val  = None
    epsilon_Jacobian_valn = None
    epsilon_Jacobian_a    = None
    the_angle_between_two = None
    the_abc_error_measure = None
    if len(approx_epsilon_lists) > 0:
        gradient_value          = last_target # batch[i-1]
        epsilon_alevel_2_real   = now_target - virtual_function(last_target).unsqueeze(0) # ltmv_true - model(last_target)
        epsilon_blevel_2_real   = (now_target - now_pred).unsqueeze(0)
        normvalue               = get_tensor_norm(approx_epsilon_lists,dim=(2,3,4))
        
        if tangent_position == 'right':
            tangent_x = last_target
        elif tangent_position == 'mid':
            tangent_x = (last_pred + last_target)/2
        elif tangent_position == 'left':
            tangent_x = last_pred
        else:
            raise NotImplementedError

        approx_epsilon_lists    = calculate_next_level_error(virtual_function, tangent_x, approx_epsilon_lists) #(N+1,B,Xdimension)
        epsilon_Jacobian_val    = get_tensor_norm(approx_epsilon_lists,dim=(2,3,4))
        epsilon_Jacobian_valn   = epsilon_Jacobian_val/normvalue
        epsilon_Jacobian_val    = epsilon_Jacobian_val[1:]
        epsilon_Jacobian_a      = epsilon_Jacobian_valn[:1] #(1,B)
        epsilon_Jacobian_valn   = epsilon_Jacobian_valn[1:] #(N,B)
        approx_epsilon_lists    = approx_epsilon_lists[1:]  #(N,B)
        epsilon_blevel_2_approx = epsilon_alevel_2_real + approx_epsilon_lists #(N,B,Xdimension) e+m(t)e
        
        the_abs_error_measure   = get_tensor_norm(epsilon_blevel_2_approx - epsilon_blevel_2_real,dim=(2,3,4))#(N,B)
        the_abc_error_measure   = get_tensor_norm(epsilon_blevel_2_approx,dim=(2,3,4)) - get_tensor_norm(epsilon_blevel_2_real,dim=(2,3,4))#(N,B)
        epsilon_blevel_v_real   = get_tensor_norm(epsilon_blevel_2_real,dim=(3,4))#(N,B,P)
        epsilon_alevel_v_real   = get_tensor_norm(epsilon_alevel_2_real,dim=(3,4))#(N,B,P)
        epsilon_blevel_v_real_norm=epsilon_blevel_v_real.mean(2)#(N,B,P) -> (N,B)
        epsilon_alevel_v_real_norm=epsilon_alevel_v_real.mean(2)#(N,B,P) -> (N,B)
        the_est_error_measure   = (get_tensor_norm(epsilon_blevel_2_real,dim=(2,3,4)) - 
                        get_tensor_norm(epsilon_alevel_2_real,dim=(2,3,4)) - 
                        get_tensor_norm(approx_epsilon_lists,dim=(2,3,4)))#(N,B)
        the_angle_between_two  = torch.einsum('bpwh,nbpwh->nb',epsilon_alevel_2_real.squeeze(0),approx_epsilon_lists
                    )/(torch.sum(epsilon_alevel_2_real**2,dim=(2,3,4)).sqrt()*torch.sum(approx_epsilon_lists**2,dim=(2,3,4)).sqrt())#(N,B)
        approx_epsilon_lists    = torch.cat([epsilon_alevel_2_real, epsilon_blevel_2_real, epsilon_blevel_2_approx]) #(N+2,B,Xdimension)
    else:
        epsilon_alevel_2_real   = (now_target - now_pred).unsqueeze(0)
        epsilon_blevel_2_real   = epsilon_alevel_2_real
        epsilon_blevel_v_real   = get_tensor_norm(epsilon_blevel_2_real,dim=(3,4))
        epsilon_alevel_v_real   = get_tensor_norm(epsilon_alevel_2_real,dim=(3,4))
        approx_epsilon_lists = torch.cat([epsilon_alevel_2_real, epsilon_blevel_2_real])#(2,B,Xdimension)
    
    return (now_target, now_pred,approx_epsilon_lists,
            the_abs_error_measure,
            the_est_error_measure,
            the_abc_error_measure,
            epsilon_alevel_v_real,
            epsilon_blevel_v_real,
            epsilon_Jacobian_val ,
            epsilon_Jacobian_valn,
            epsilon_Jacobian_a,
            the_angle_between_two    )




def run_one_fourcast_iter(model, batch, idxes, fourcastresult,dataset,
                    save_prediction_first_step=None,save_prediction_final_step=None,
                    snap_index=None,do_error_propagration_monitor=False):
    if dataset.dataset_flag=='2D70N':
        model.fourcast_for_2D70N = True
        
    accu_series=[]
    rmse_series=[]
    rmse_maps = []
    hmse_series=[]
    extra_info = {}
    time_step_1_mode=False
    batch_variance_line_pred = [] 
    batch_variance_line_true = []
    the_abs_error_measure_list = []
    the_est_error_measure_list = []
    the_abc_error_measure_list = []
    epsilon_alevel_v_real_list = []
    epsilon_blevel_v_real_list = []
    epsilon_Jacobian_val_list = []
    epsilon_Jacobian_valn_list = []
    epsilon_Jacobian_a_list = []
    the_angle_between_two_list = []
    # we will also record the variance for each slot, assume the input is a time series of data
    # [ (B, P, W, H) -> (B, P, W, H) -> .... -> (B, P, W, H) ,(B, P, W, H)]
    # we would record the variance of each batch on the location (W,H)
    clim = model.clim
    
    start = batch[0:model.history_length] # start must be a list    
    
    snap_line = []
    if (snap_index is not None) and (0 not in [len(t) for t in snap_index]):  
        for i,tensor in enumerate(start):
            # each tensor is like (B, 70, 32, 64) or (B, P, Z, W, H)
            snap_line.append([len(snap_line), get_tensor_value(tensor,snap_index, time=model.history_length),'input'])
    
    approx_epsilon_lists = []
    last_pred = last_target = None
    for i in range(model.history_length,len(batch), model.pred_len):# i now is the target index
        end = batch[i:i+model.pred_len]
        end = end[0] if len(end) == 1 else end
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,end,dataset,time_step_1_mode)
        if 'Delta' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.combine_base_delta(start[-1][1], start[-1][0]) 
                target    = dataset.combine_base_delta(      end[1],       end[0])  
        elif 'deseasonal' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.addseasonal(start[-1][1], start[-1][0])
                target    = dataset.addseasonal(end[1], end[0])
        elif '68pixelnorm' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.recovery(start[-1])
                target    = dataset.recovery(end)
                 
        for extra_info_from_model in extra_info_from_model_list:
            for key, val in extra_info_from_model.items():
                if i not in extra_info:extra_info[i] = {}
                if key not in extra_info[i]:extra_info[i][key] = []
                extra_info[i][key].append(val)

        ltmv_trues = dataset.inv_normlize_data([target])[0]#.detach().cpu()
        ltmv_preds = ltmv_pred#.detach().cpu()

        if do_error_propagration_monitor and len(approx_epsilon_lists) < 9: # only do this 8 times
            tangent_position = 'mid'
            (last_target, last_pred,approx_epsilon_lists,
                the_abs_error_measure,the_est_error_measure,the_abc_error_measure,
                epsilon_alevel_v_real,epsilon_blevel_v_real,epsilon_Jacobian_val,epsilon_Jacobian_valn,epsilon_Jacobian_a,the_angle_between_two
            ) = get_error_propagation(last_pred, last_target, ltmv_trues, ltmv_preds, model ,approx_epsilon_lists, tangent_position=tangent_position)
            if the_abs_error_measure is not None: the_abs_error_measure_list.append(the_abs_error_measure.detach().cpu())
            if the_est_error_measure is not None: the_est_error_measure_list.append(the_est_error_measure.detach().cpu())
            if the_abc_error_measure is not None: the_abc_error_measure_list.append(the_abc_error_measure.detach().cpu())
            if epsilon_alevel_v_real is not None: epsilon_alevel_v_real_list.append(epsilon_alevel_v_real.detach().cpu())
            if epsilon_blevel_v_real is not None: epsilon_blevel_v_real_list.append(epsilon_blevel_v_real.detach().cpu())
            if epsilon_Jacobian_val  is not None: epsilon_Jacobian_val_list.append(epsilon_Jacobian_val.detach().cpu())
            if epsilon_Jacobian_valn is not None: epsilon_Jacobian_valn_list.append(epsilon_Jacobian_valn.detach().cpu())
            if epsilon_Jacobian_a is not None: epsilon_Jacobian_a_list.append(epsilon_Jacobian_a.detach().cpu())
            if the_angle_between_two is not None: the_angle_between_two_list.append(the_angle_between_two.detach().cpu())
        if model.pred_len > 1:
            ltmv_trues = ltmv_trues.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
            ltmv_preds = ltmv_preds.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)\
            time_list  = range(i,i+model.pred_len)
        else:
            ltmv_trues = ltmv_trues.unsqueeze(0)
            ltmv_preds = ltmv_preds.unsqueeze(0)
            time_list  = [i]

        if save_prediction_first_step is not None and i==model.history_length:save_prediction_first_step[idxes] = ltmv_pred.detach().cpu()
        if save_prediction_final_step is not None and i==len(batch) - 1:save_prediction_final_step[idxes] = ltmv_pred.detach().cpu()

        for j,(ltmv_true,ltmv_pred) in enumerate(zip(ltmv_trues,ltmv_preds)):
            time = time_list[j]
            ### enter CPU computing
            if len(ltmv_true.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_true = ltmv_true.flatten(1,2) 
            if len(ltmv_pred.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_pred = ltmv_pred.flatten(1,2) 
            if len(clim.shape)!=len(ltmv_pred.shape):
                ltmv_pred = ltmv_pred.squeeze(-1)
                ltmv_true = ltmv_true.squeeze(-1) # temporary use this for timestamp input like [B, P, w,h,T]
            
            if snap_index is not None:
                snap_line.append([time, get_tensor_value(ltmv_pred,snap_index, time=time),'pred'])
                snap_line.append([time, get_tensor_value(ltmv_true,snap_index, time=time),'true'])
            
            statistic_dim = tuple(range(2,len(ltmv_true.shape))) # always assume (B,P,Z,W,H)
            batch_variance_line_pred.append(ltmv_pred.std(dim=statistic_dim).detach().cpu())
            batch_variance_line_true.append(ltmv_true.std(dim=statistic_dim).detach().cpu())
            #accu_series.append(compute_accu(ltmv_pred, ltmv_true ).detach().cpu())
            accu_series.append(compute_accu(ltmv_pred - clim.to(ltmv_pred.device), ltmv_true - clim.to(ltmv_pred.device)).detach().cpu())
            rmse_v,rmse_map = compute_rmse(ltmv_pred , ltmv_true, return_map_also=True)
            rmse_series.append(rmse_v.detach().cpu()) #(B,70)
            rmse_maps.append(rmse_map.detach().cpu()) #(70,32,64)
            hmse_value = compute_rmse(ltmv_pred[...,8:24,:], ltmv_true[...,8:24,:]) if ltmv_pred.shape[-2] == 32 else -torch.ones_like(rmse_v)
            hmse_series.append(hmse_value.detach().cpu())
        #torch.cuda.empty_cache()

    

    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,property_num)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,property_num)
    hmse_series = torch.stack(hmse_series,1) # (B,fourcast_num,property_num)
    batch_variance_line_pred = torch.stack(batch_variance_line_pred,1) # (B,fourcast_num,property_num)
    batch_variance_line_true = torch.stack(batch_variance_line_true,1) # (B,fourcast_num,property_num)
    
    for idx, accu,rmse,hmse, std_pred,std_true in zip(idxes,accu_series,rmse_series,hmse_series,
                                                batch_variance_line_pred,batch_variance_line_true):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'accu':accu,"rmse":rmse,'std_pred':std_pred,'std_true':std_true,'snap_line':[],
                                      "hmse":hmse}
    
    if len(the_abs_error_measure_list) > 0 :
        # `the_abs_error_measure_list` is a list of [    (1,B) , (2,B), (3,B), ..., (8,B) ]
        the_abs_error_measure_list = torch.cat(the_abs_error_measure_list).permute(1,0) #(36, B) -> (B, 36)
        the_est_error_measure_list = torch.cat(the_est_error_measure_list).permute(1,0) #(36, B) -> (B, 36)
        the_angle_between_two_list = torch.cat(the_angle_between_two_list).permute(1,0) #(36, B) -> (B, 36)
        the_abc_error_measure_list = torch.cat(the_abc_error_measure_list).permute(1,0) #(36, B) -> (B, 36)
        # `epsilon_Jacobian_val_list` is also a list of [    (1,B) , (2,B), (3,B), ..., (8,B) ]
        epsilon_Jacobian_val_list  = torch.cat(epsilon_Jacobian_val_list).permute(1,0)
        epsilon_Jacobian_valn_list = torch.cat(epsilon_Jacobian_valn_list).permute(1,0)
        # `epsilon_blevel_v_real_list` is a list of values [ (1,B,P) (1,B,P) (1,B), ... , (1,B)]
        epsilon_blevel_v_real_list = torch.cat(epsilon_blevel_v_real_list).permute(1,0,2)#(B, L, P)
        epsilon_alevel_v_real_list = torch.cat(epsilon_alevel_v_real_list).permute(1,0,2)#(B, L, P)
        epsilon_Jacobian_a_list    = torch.cat(epsilon_Jacobian_a_list).permute(1,0)#
        for idx, abs_error, est_error,abc_error,epsilon_alevel_v,epsilon_blevel_v,epsilon_Jacobian,epsilon_Jacobian_norm,epsilon_Jacobian_A_norm,the_angle_between_two in zip(idxes,
                the_abs_error_measure_list,the_est_error_measure_list,the_abc_error_measure_list,
                epsilon_alevel_v_real_list,epsilon_blevel_v_real_list,
                epsilon_Jacobian_val_list,epsilon_Jacobian_valn_list,epsilon_Jacobian_a_list,
                the_angle_between_two_list):
            fourcastresult[idx.item()]['abs_error'] = abs_error
            fourcastresult[idx.item()]['est_error'] = est_error
            fourcastresult[idx.item()]['abc_error'] = abc_error
            fourcastresult[idx.item()]['alevel_v'] = epsilon_alevel_v
            fourcastresult[idx.item()]['blevel_v'] = epsilon_blevel_v
            fourcastresult[idx.item()]['Jacobianv'] = epsilon_Jacobian
            fourcastresult[idx.item()]['JacobV_N'] = epsilon_Jacobian_norm
            fourcastresult[idx.item()]['JacobVA_N'] = epsilon_Jacobian_A_norm
            fourcastresult[idx.item()]['Angle'] = the_angle_between_two
            
    if snap_index is not None:
        for batch_id,select_batch_id in enumerate(snap_index[0]):
            for snap_each_fourcast_time in snap_line:
                # each snap is tensor (b, p, L)
                time_step, tensor, label = snap_each_fourcast_time
                fourcastresult[idxes[select_batch_id].item()]['snap_line'].append([
                    time_step,tensor[batch_id],label
                ])
            #  (p, L) -> (p, L) -> (p, L)
    if "global_rmse_map" not in fourcastresult:
        fourcastresult['global_rmse_map'] = rmse_maps # it is a map list (70,32,64) -> (70, 28,64) ->....
    else:
        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],rmse_maps)]
    return fourcastresult,extra_info

def run_one_fourcast_iter_with_history(model, start, batch, idxes, fourcastresult):
    accu_series=[]
    rmse_series=[]
    out = start
    extra_info = {}
    history_sum_true = history_sum_pred = batch[0].permute(0,2,3,1) if batch[0].shape[1]!=20 or batch[0].shape[1]!=12 else batch[0]
    for i in range(1,len(batch)):
        out   = model(out)
        extra_loss = 0
        if isinstance(out,(list,tuple)):
            extra_loss=out[1]
            for extra_info_from_model in out[2:]:
                for key, val in extra_info_from_model.items():
                    if i not in extra_info:extra_info[i] = {}
                    if key not in extra_info[i]:extra_info[i][key] = []
                    extra_info[i][key].append(val)
            out = out[0]
        ltmv_pred = out.permute(0,2,3,1)# (B, P, W, H ) -> # (B, W, H, P)
        ltmv_true = batch[i].permute(0,2,3,1)# (B, P, W, H ) -> # (B, W, H, P)
        history_sum_pred+=ltmv_pred
        history_sum_true+=ltmv_true
        history_mean_pred=history_sum_pred/(i+1)
        history_mean_true=history_sum_true/(i+1)
        ltmsv_pred = ltmv_pred - history_mean_pred
        ltmsv_true = ltmv_true - history_mean_true
        accu_series.append(compute_accu(ltmsv_pred, ltmsv_true).detach().cpu())
        rmse_series.append(compute_rmse(ltmv_pred , ltmv_true ).detach().cpu())
    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,20)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,20)
    for idx, accu,rmse in zip(idxes,accu_series,rmse_series):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'accu':accu,"rmse":rmse}
    return fourcastresult,extra_info

def fourcast_step(data_loader, model,logsys,random_repeat = 0,snap_index=None,do_error_propagration_monitor=False):
    model.eval()
    logsys.eval()
    status     = 'test'
    gpu        = dist.get_rank() if hasattr(model,'module') else 0
    Fethcher   = Datafetcher
    prefetcher = Fethcher(data_loader,next(model.parameters()).device)
    batches = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    device = next(model.parameters()).device
    data_cost = train_cost = rest_cost = 0
    now = time.time()
    model.clim = torch.Tensor(data_loader.dataset.clim_tensor).to(device)
    fourcastresult={}
    save_prediction_first_step = None#torch.zeros_like(data_loader.dataset.data)
    save_prediction_final_step = None#torch.zeros_like(data_loader.dataset.data)
    # = 100
    intervel = batches//logsys.log_trace_times + 1

    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            #if inter_b.now>10:break
            data_cost += time.time() - now;now = time.time()
            step        = inter_b.now
            idxes,batch = prefetcher.next()
            batch       = make_data_regular(batch,half_model)
            # first sum should be (B, P, W, H )
            the_snap_index_in_iter = None
            if snap_index is not None:
                select_start_timepoints = snap_index[0]
                the_snap_index_in_iter=[[],snap_index[1],snap_index[2]]
                the_snap_index_in_iter[0] = [batch_id for batch_id, idx in enumerate(idxes) if idx in select_start_timepoints]
                if len(the_snap_index_in_iter[0]) == 0: the_snap_index_in_iter=None
            #if the_snap_index_in_iter is None:continue
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                fourcastresult,extra_info = run_one_fourcast_iter(model, batch, idxes, fourcastresult,data_loader.dataset,
                                         save_prediction_first_step=save_prediction_first_step,
                                         save_prediction_final_step=save_prediction_final_step,
                                         snap_index=the_snap_index_in_iter,do_error_propagration_monitor=do_error_propagration_monitor)
            train_cost += time.time() - now;now = time.time()
            for _ in range(random_repeat):
                raise NotImplementedError
                fourcastresult,extra_info = run_one_fourcast_iter(model, [batch[0]*(1 + torch.randn_like(global_start)*0.05)]+batch[1:], idxes, fourcastresult,data_loader.dataset)
            
            rest_cost += time.time() - now;now = time.time()
            if (step+1) % intervel==0 or step==0:
                for idx, val_pool in extra_info.items():
                    for key, val in val_pool.items():
                        logsys.record(f'test_{key}_each_fourcast_step', np.mean(val), idx, epoch_flag = 'time_step')
                outstring=(f"epoch:fourcast iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}] cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} [Rest]:{rest_cost/intervel:.1e}")
                inter_b.lwrite(outstring, end="\r")
            #if inter_b.now >2:break
    if save_prediction_first_step is not None:torch.save(save_prediction_first_step,os.path.join(logsys.ckpt_root,'save_prediction_first_step')) 
    if save_prediction_final_step is not None:torch.save(save_prediction_final_step,os.path.join(logsys.ckpt_root,'save_prediction_final_step')) 
    fourcastresult['snap_index'] = snap_index
    return fourcastresult

def create_fourcast_metric_table(fourcastresult, logsys,test_dataset,collect_names=['500hPa_geopotential','850hPa_temperature']):
    prefix_pool={
        'only_backward':"time_reverse_",
        'only_forward':""
    }
    prefix = prefix_pool[test_dataset.time_reverse_flag]

    if isinstance(fourcastresult,str):
        # then it is the fourcastresult path
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'fourcastresult.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val
                else:
                    if key == 'global_rmse_map':

                        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],tmp['global_rmse_map'])]
                    else:
                        fourcastresult[key] = val # overwrite
        

    property_names = test_dataset.vnames
    if hasattr(test_dataset,"pred_channel_for_next_stamp"):
        property_names = [property_names[t] for t in test_dataset.pred_channel_for_next_stamp]
        test_dataset.unit_list = test_dataset.unit_list[test_dataset.pred_channel_for_next_stamp]
    # if 'UVTP' in args.wrapper_model:
    #     property_names = [property_names[t] for t in eval(args.wrapper_model).pred_channel_for_next_stamp]
    ## <============= ACCU ===============>
    if fourcastresult['snap_index'] is None:del fourcastresult['snap_index']
    accu_list = torch.stack([p['accu'].cpu() for p in fourcastresult.values() if 'accu' in p]).numpy()    
    total_num = len(accu_list)
    accu_list = accu_list.mean(0)# (fourcast_num,property_num)
    real_times = [(predict_time+1)*test_dataset.time_intervel*test_dataset.time_unit for predict_time in range(len(accu_list))]
    save_and_log_table(accu_list,logsys, prefix+'accu_table', property_names, real_times)    

    ## <============= RMSE ===============>
    rmse_list = torch.stack([p['rmse'].cpu() for p in fourcastresult.values() if 'rmse' in p]).mean(0)# (fourcast_num,property_num)
    save_and_log_table(rmse_list,logsys, prefix+'rmse_table', property_names, real_times)       
    
    
    ## <============= HMSE ===============>
    hmse_list = torch.stack([p['hmse'].cpu() for p in fourcastresult.values() if 'hmse' in p]).mean(0)# (fourcast_num,property_num)
    if (hmse_list>0).all():
        save_and_log_table(rmse_list,logsys, prefix+'hmse_table', property_names, real_times)           
    
    ## <============= Error_Norm ===============>
    #fourcastresult[idx.item()]['abs_error'] = abs_error
    #fourcastresult[idx.item()]['est_error'] = est_error
    
    ## <============= STD_Location ===============>
    meanofstd = torch.stack([p['std_pred'].cpu() for p in fourcastresult.values() if 'std_pred' in p]).numpy().mean(0)# (B, (fourcast_num,property_num)
    save_and_log_table(meanofstd,logsys, prefix+'meanofstd_table', property_names, real_times)       

    stdofstd = torch.stack([p['std_pred'].cpu() for p in fourcastresult.values() if 'std_pred' in p]).numpy().std(0)# (B, (fourcast_num,property_num)
    save_and_log_table(stdofstd,logsys, prefix+'stdofstd_table', property_names, real_times)      

    
    try:
        if not isinstance(test_dataset.unit_list,int):
            unit_list = torch.Tensor(test_dataset.unit_list).to(rmse_list.device)
            #print(unit_list)
            unit_num  = max(unit_list.shape)
            unit_num  = len(property_names)
            unit_list = unit_list.reshape(1,unit_num)
            property_num = len(property_names)
            if property_num > unit_num:
                assert property_num%unit_num == 0
                unit_list = torch.repeat_interleave(unit_list,int(property_num//unit_num),dim=1)
        else:
            logsys.info(f"unit list is int, ")
            unit_list= test_dataset.unit_list
        
        rmse_unit_list= (rmse_list*unit_list)
        save_and_log_table(rmse_unit_list,logsys, prefix+'rmse_unit_list', property_names, real_times)
        hmse_unit_list= (hmse_list*unit_list)
        if (hmse_list>0).all():
            save_and_log_table(hmse_unit_list,logsys, prefix+'hmse_unit_list', property_names, real_times)       
    except:
        logsys.info(f"get wrong when use unit list, we will fource let [rmse_unit_list] = [rmse_list]")
        traceback.print_exc()

    ## <============= Snap_PLot ==================>
    snap_tables = []
    if ('snap_index' in fourcastresult) and (fourcastresult['snap_index'] is not None):
        snap_index = fourcastresult['snap_index']
        select_snap_start_time_point = snap_index[0]
        select_snap_show_property_id = snap_index[1]
        select_snap_show_location    = snap_index[2]
        select_snap_property_name    = [property_names[iidd] for iidd in select_snap_show_property_id]
        for select_time_point in select_snap_start_time_point:
            timestamp = test_dataset.datatimelist_pool['test'][select_time_point]
            if select_time_point in fourcastresult: # in case do not record
                linedata = fourcastresult[select_time_point]['snap_line']
                for predict_time_point, tensor, label in linedata:
                    predict_timestamp = (predict_time_point)*test_dataset.time_intervel*test_dataset.time_unit
                    #  TENSOR --> (P,N) 
                    for propery_name, property_along_location in zip(select_snap_property_name,tensor):
                        for pos_id, value in enumerate(property_along_location):
                            
                            location_x = select_snap_show_location[0][pos_id]
                            location_y = select_snap_show_location[1][pos_id]
                            snap_tables.append([timestamp, label, predict_timestamp,propery_name,location_x,location_y,value])
                    
        logsys.add_table("snap_table", snap_tables , 0, ['start_time',"label","predict_time","propery","pos_x","pos_y","value"])

    if 'global_rmse_map' in fourcastresult:
        global_rmse_map = fourcastresult['global_rmse_map']
        mean_global_rmse_map = [torch.sqrt(t/total_num) for t in global_rmse_map]
        for j,prop_name in enumerate(property_names): 
            if prop_name not in collect_names:continue
            big_step = len(mean_global_rmse_map)
            vmin = min([map_per_time[...,j].min().item() for map_per_time in mean_global_rmse_map])
            vmax = max([map_per_time[...,j].max().item() for map_per_time in mean_global_rmse_map])
            for i,map_per_time in enumerate(mean_global_rmse_map):
                name = f"global_rmse_map_for_{prop_name}"
                s_dir= os.path.join(logsys.ckpt_root,"figures")
                if not os.path.exists(s_dir):os.makedirs(s_dir)
                real_time = real_times[i]
                spath= os.path.join(s_dir,name+f'.{real_time}h.png')
                data = map_per_time[...,j]
                if data.shape[-2] < 32:
                    pad = (32 - data.shape[-2])//2
                    data= torch.nn.functional.pad(data,(0,0,pad,pad),'constant',100)
                assert data.shape[-2] == 32
                #images = wandb.Image(mean_global_rmse_map[i][...,j], caption='rmse_map')
                plt.imshow(data.numpy(),vmin=vmin,vmax=vmax,cmap='gray')
                plt.title(f"value range: {vmin:.3f}-{vmax:.3f}")
                plt.xticks([]);plt.yticks([])
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:wandb.Image(spath)})
                

    info_pool_list = []
    for predict_time in range(len(accu_list)):
        real_time = real_times[predict_time]
        info_pool={}
        accu_table = accu_list[predict_time]
        rmse_table = rmse_list[predict_time]
        rmse_table_unit = rmse_unit_list[predict_time]

        hmse_table = hmse_list[predict_time]
        hmse_table_unit = hmse_unit_list[predict_time]
        for name,accu,rmse,rmse_unit,hmse,hmse_unit in zip(property_names,accu_table,rmse_table, rmse_table_unit,hmse_table, hmse_table_unit):
            
            info_pool[prefix + f'test_accu_{name}'] = accu.item()
            info_pool[prefix + f'test_rmse_{name}'] = rmse.item()
            info_pool[prefix + f'test_rmse_unit_{name}'] = rmse_unit.item()
            info_pool[prefix + f'test_hmse{name}']    = hmse.item()
            info_pool[prefix + f'test_hmse_unit_{name}'] = hmse_unit.item()
            if real_time in [12, 24, 48, 72, 96, 120]:  
                if name not in collect_names:continue      
                info_pool[prefix + f'{real_time}_hours_test_rmse_unit_{name}'] = rmse_unit.item()
                info_pool[prefix + f'{real_time}_hours_test_rmse_{name}'] = rmse.item()
                info_pool[prefix + f'{real_time}_hours_test_hmse_{name}'] = hmse.item()
                info_pool[prefix + f'{real_time}_hours_test_hmse_unit_{name}'] = hmse_unit.item()
        info_pool['real_time'] = real_time
        for key, val in info_pool.items():
            logsys.record(key,val, predict_time, epoch_flag = 'time_step')
        info_pool_list.append(info_pool)

    

    return info_pool_list

def run_fourcast(args, model,logsys,test_dataloader=None):
    import warnings
    warnings.filterwarnings("ignore")
    logsys.info_log_path = os.path.join(logsys.ckpt_root, 'fourcast.info')
    
    if test_dataloader is None:
        test_dataset,  test_dataloader = get_test_dataset(args)

    test_dataset = test_dataloader.dataset
    

    #args.force_fourcast=True
    gpu       = dist.get_rank() if hasattr(model,'module') else 0
    fourcastresult_path = os.path.join(logsys.ckpt_root,f"fourcastresult.gpu_{gpu}")
    if not os.path.exists(fourcastresult_path) or  args.force_fourcast:
        logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
        logsys.info("starting fourcast~!")
        with open(os.path.join(logsys.ckpt_root,'weight_path'),'w') as f:f.write(args.pretrain_weight)
        fourcastresult  = fourcast_step(test_dataloader, model,logsys,
                                    random_repeat = args.fourcast_randn_initial,
                                    snap_index=args.snap_index,do_error_propagration_monitor=args.do_error_propagration_monitor)
        torch.save(fourcastresult,fourcastresult_path)
        logsys.info(f"save fourcastresult at {fourcastresult_path}")
    else:
        logsys.info(f"load fourcastresult at {fourcastresult_path}")
        fourcastresult = torch.load(fourcastresult_path)

    if not args.distributed:
        create_fourcast_metric_table(fourcastresult, logsys,test_dataset)
    
    return 1


#########################################
######## nodal snape forward step #######
#########################################

def donoting():
    pass

def run_nodaloss_snap(model, batch, idxes, fourcastresult,dataset, property_select = [38,49],chunk_size=1024):
    time_step_1_mode=False
    L1meassures  = []
    L2meassures  = []
    L1meassure   = L2meassure = torch.zeros(1)
    start = batch[0:model.history_length] # start must be a list

    with torch.no_grad():
        for i in range(model.history_length,len(batch)):# i now is the target index
            func_model = lambda x:model(x)[:,property_select]
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,batch[i],dataset,time_step_1_mode)
                now = time.time()
                L1meassure = vmap(the_Nodal_L1_meassure(func_model), (0))(start[-1].unsqueeze(1)) # (B, Pick, W,H)
                L2meassure = vmap(the_Nodal_L2_meassure(func_model,chunk_size=chunk_size), (0))(start[-1].unsqueeze(1))# (B, Pick, W,H)
                print(f"step_{i:3d} L2 computing finish, cost:{time.time() - now }") 
                L1meassures.append(L1meassure.detach().cpu())
                L2meassures.append(L2meassure.detach().cpu())

    L1meassures = torch.stack(L1meassures,1) # (B, fourcast_num, Pick_property_num, W,H)
    L2meassures = torch.stack(L2meassures,1) # (B, fourcast_num, Pick_property_num, W,H)

    for idx, L1meassure,L2meassure in zip(idxes,L1meassures,L2meassures):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'L1meassure':L1meassure,"L2meassure":L2meassure}
    return fourcastresult

def snap_nodal_step(data_loader, model,logsys, property_select = [38,49],batch_limit=1,chunk_size=1024):
    model.eval()
    logsys.eval()
    status     = 'test'
    gpu        = dist.get_rank() if hasattr(model,'module') else 0
    Fethcher   = Datafetcher
    prefetcher = Fethcher(data_loader,next(model.parameters()).device)
    batches = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    device = next(model.parameters()).device
    data_cost = train_cost = rest_cost = 0
    now = time.time()
    model.clim = torch.Tensor(data_loader.dataset.clim_tensor).to(device)
    fourcastresult={}
    intervel = batches//100 + 1
    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            step           = inter_b.now
            idxes,batch    = prefetcher.next()
            batch          = make_data_regular(batch,half_model)
            fourcastresult = run_nodaloss_snap(model, batch, idxes, fourcastresult,data_loader.dataset, 
                                               property_select = property_select,chunk_size=chunk_size)
            if (step+1) % intervel==0 or step==0:
                outstring=(f"epoch:fourcast iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}]")
                inter_b.lwrite(outstring, end="\r")
            if inter_b.now > batch_limit:break
    return fourcastresult

def run_nodalosssnap(args, model,logsys,test_dataloader=None,property_select=[38,49]):
    import warnings
    warnings.filterwarnings("ignore")
    
    if test_dataloader is None:test_dataset,  test_dataloader = get_test_dataset(args)
    test_dataset = test_dataloader.dataset
    logsys.info_log_path = os.path.join(logsys.ckpt_root, f'nodal_snap_on_{test_dataset.split}_dataset.info')
    #args.force_fourcast=True
    gpu       = dist.get_rank() if hasattr(model,'module') else 0
    fourcastresult_path = os.path.join(logsys.ckpt_root,f"nodal_snap_on_{test_dataset.split}_dataset.gpu_{gpu}")

    if not os.path.exists(fourcastresult_path) or  args.force_fourcast:
        logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
        logsys.info("starting fourcast~!")
        fourcastresult  = snap_nodal_step(test_dataloader, model,logsys, property_select = property_select,
                                          batch_limit=args.batch_limit,chunk_size=args.chunk_size)
        fourcastresult['property_select'] = property_select
        torch.save(fourcastresult,fourcastresult_path)
        logsys.info(f"save fourcastresult at {fourcastresult_path}")
    else:
        logsys.info(f"load fourcastresult at {fourcastresult_path}")
        fourcastresult = torch.load(fourcastresult_path)

    if not args.distributed:
        create_nodal_loss_snap_metric_table(fourcastresult, logsys,test_dataset)
    
    return 1

def create_nodal_loss_snap_metric_table(fourcastresult, logsys,test_dataset):
    prefix_pool={
        'only_backward':"time_reverse_",
        'only_forward':""
    }
    prefix = prefix_pool[test_dataset.time_reverse_flag]

    if isinstance(fourcastresult,str):
        # then it is the fourcastresult path
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'nodal_snap_on_test_dataset.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val

    property_names = test_dataset.vnames
    property_select= fourcastresult['property_select']
    
    L1meassures = torch.stack([p['L1meassure'].cpu() for p in fourcastresult.values() if 'L1meassure' in p]) #(B, fourcast_num, Pick_property_num, W,H)
    L2meassures = torch.stack([p['L2meassure'].cpu() for p in fourcastresult.values() if 'L2meassure' in p]) #(B, fourcast_num, Pick_property_num, W,H)
    if len(L1meassures.shape)==6 and L1meassures.shape[2]==1:L1meassures=L1meassures[:,:,0]
    if len(L2meassures.shape)==6 and L2meassures.shape[2]==1:L2meassures=L2meassures[:,:,0]

    print(L1meassures.shape)
    print(L2meassures.shape)

    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        if torch.isnan(meassure).any() or torch.isinf(meassure).any():
            print(f"{metric_name}meassure has nan of inf")

    select_keys = [k for k in fourcastresult.keys() if isinstance(k,int)]

    print("create L1/L2 loss .................")
    # the first thing is to record the L1 measure per (B,P,W,H) per fourcast_num
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for fourcast_step, tensor_per_property in enumerate(meassure.permute(1,2,0,3,4).flatten(-3,-1)):
            for property_id, tensor in enumerate(tensor_per_property):
                property_name = property_names[property_select[property_id]]
                value = torch.mean((tensor - 1)**2)
                if torch.isnan(value):
                    print(f"{metric_name}_loss_for_{property_name}_with_{len(meassure)}_batches_on_{test_dataset.split}_at_{fourcast_step}_step_is bad")
                logsys.wandblog({f"{metric_name}_loss_for_{property_name}_with_{len(meassure)}_batches_on_{test_dataset.split}":value,'time_step':fourcast_step})     

    print("create mean std .................")
    # the first thing is to record the L1 measure per (B,P,W,H) per fourcast_num
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        table = []
        for fourcast_step, tensor_per_property in enumerate(meassure.permute(1,2,0,3,4).flatten(-3,-1)):
            for property_id, tensor in enumerate(tensor_per_property):
                property_name = property_names[property_select[property_id]]
                table.append([fourcast_step,property_name, idx, tensor.mean().item(),tensor.std().item()])
        logsys.add_table(f"{metric_name}_table", table , 0, ['fourcast_step',"property","idx","mean","std"])     

    print("create histgram .................")
    s_dir= os.path.join(logsys.ckpt_root,"figures")
    if not os.path.exists(s_dir):os.makedirs(s_dir)
    ## then we are going to plot the histgram
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for property_id, tensor_per_property in enumerate(meassure.permute(2,1,0,3,4).flatten(-3,-1)):
            property_name = property_names[property_select[property_id]]
            x_min = tensor_per_property.min()
            x_max = tensor_per_property.max()
            for fourcast_step, tensor in enumerate(tensor_per_property):
                
                data = tensor.numpy()
                name = f'{metric_name}_histogram_{property_name}_{len(meassure)}'
                if fourcast_step ==0:
                    table = wandb.Table(data=data[:,None], columns=[metric_name])
                    wandb.log({name+f"_at_time_step_{fourcast_step}": wandb.plot.histogram(table, metric_name,
                        title=f"{metric_name} histram for {property_name} with {len(meassure)} batches"),'time_step':fourcast_step})
                fig = plt.figure()
                smoothhist(data)
                plt.xlim(x_min,x_max)
                spath= os.path.join(s_dir,name+f'.step{fourcast_step}.png')
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:fig})
    ## then we are going to plot the heatmap
    s_dir= os.path.join(logsys.ckpt_root,"figures")
    if not os.path.exists(s_dir):os.makedirs(s_dir)
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for property_id, tensor_per_property in enumerate(meassure.permute(2,1,0,3,4)):
            property_name = property_names[property_select[property_id]]
            x_min = 0
            x_max = 1.2
            for fourcast_step, tensor in enumerate(tensor_per_property):
                # tensor is (B, W, H), we only pick the first 
                the_map = tensor[0]
                #print(the_map.shape)
                start_time = select_keys[0]
                vmin = the_map.min()
                vmax = the_map.max()
                name = f"{metric_name}_map_{property_name}_start_from_{start_time}"
                spath= os.path.join(s_dir,name+f'.step{fourcast_step}.png')
                plt.imshow(the_map.numpy(),vmin=vmin,vmax=vmax,cmap='gray')
                plt.title(f"value range: {vmin:.3f}-{vmax:.3f}")
                plt.xticks([]);plt.yticks([])
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:wandb.Image(spath)})
                
    return


def seed_worker(worker_id):# Multiprocessing randomnes for multiGPU train #https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_and_valid_dataset(args,train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    dataset_type   = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    
    train_dataset  = dataset_type(split="train" if not args.debug else 'test',dataset_tensor=train_dataset_tensor,
                                  record_load_tensor=train_record_load,**args.dataset_kargs)
    val_dataset   = dataset_type(split="valid" if not args.debug else 'test',dataset_tensor=valid_dataset_tensor,
                                  record_load_tensor=valid_record_load,**args.dataset_kargs)

    train_datasampler = DistributedSampler(train_dataset, shuffle=args.do_train_shuffle) if args.distributed else None
    val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader  = DataLoader(train_dataset, args.batch_size,   sampler=train_datasampler, num_workers=args.num_workers, pin_memory=True,
                                   drop_last=True,worker_init_fn=seed_worker,generator=g,shuffle=True if ((not args.distributed) and args.do_train_shuffle) else False)
    val_dataloader    = DataLoader(val_dataset  , args.valid_batch_size, sampler=val_datasampler,   num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return   train_dataset,   val_dataset, train_dataloader, val_dataloader

def get_test_dataset(args,test_dataset_tensor=None,test_record_load=None):
    time_step = args.time_step if "fourcast" in args.mode else 5*24//6 + args.time_step
    dataset_kargs = copy.deepcopy(args.dataset_kargs)
    dataset_kargs['time_step'] = time_step
    if dataset_kargs['time_reverse_flag'] in ['only_forward','random_forward_backward']:
        dataset_kargs['time_reverse_flag'] = 'only_forward'
    dataset_type = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    #print(dataset_kargs)
    split = args.split if hasattr(args,'split') and args.split else "test"
    test_dataset = dataset_type(split=split, with_idx=True,dataset_tensor=test_dataset_tensor,record_load_tensor=test_record_load,**dataset_kargs)
    if args.wrapper_model and hasattr(eval(args.wrapper_model),'pred_channel_for_next_stamp'):
        test_dataset.pred_channel_for_next_stamp = eval(args.wrapper_model).pred_channel_for_next_stamp
    
    assert hasattr(test_dataset,'clim_tensor')
    test_datasampler  = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
    test_dataloader   = DataLoader(test_dataset, args.valid_batch_size, sampler=test_datasampler, num_workers=args.num_workers, pin_memory=False)
    
    
    return   test_dataset,   test_dataloader


#########################################
######## argument and config set #######
#########################################

def tuple2str(_tuple):
    if isinstance(_tuple,(list,tuple)):
        return '.'.join([str(t) for t in (_tuple)])
    else:
        return _tuple

def get_model_name(args):
    model_name = args.model_type
    if "FED" in args.model_type:
        mode_name =args.modes.replace(",","-")
        return f"{args.model_type}.{args.mode_select}_select.M{mode_name}_P{args.pred_len}L{args.label_len}"
    if "AFN" in args.model_type and hasattr(args,'model_depth') and args.model_depth == 6:
        model_name = "small_" + model_name
    model_name = f"ViT_in_bulk-{model_name}" if len(args.img_size)>2 else model_name
    model_name = f"{args.wrapper_model}-{model_name}" if args.wrapper_model else model_name
    model_name = f"{model_name}_Patch_{tuple2str(args.patch_range)}" if (args.patch_range and 'Patch' in args.dataset_type) else model_name
    return model_name

def get_datasetname(args):
    datasetname = args.dataset_type
    if not args.dataset_type and args.train_set in train_set:
        datasetname = train_set[args.train_set][4].__name__
    if not datasetname:
        raise NotImplementedError("please use right dataset type")
        
    if datasetname in ["",'ERA5CephDataset','ERA5CephSmallDataset']:
        datasetname  = "ERA5_20-12" if 'physics' in args.train_set else "ERA5_20"
    return datasetname

def get_projectname(args):
    model_name   = get_model_name(args)
    datasetname  = get_datasetname(args)

    if "Self" in datasetname:
        property_names = 'UVTPH'
        picked_input_name = "".join([property_names[t] for t in args.picked_input_property])
        picked_output_name= "".join([property_names[t] for t in args.picked_output_property])
        project_name = f"{picked_input_name}_{picked_output_name}"
    else:
        project_name = f"{args.mode}-{args.train_set}"
        if hasattr(args,'random_time_step') and args.random_time_step:project_name = 'rd_sp_'+project_name 
        if hasattr(args,'time_step') and args.time_step:              project_name = f"ts_{args.time_step}_" +project_name 
        if hasattr(args,'history_length') and args.history_length !=1:project_name = f"his_{args.history_length}_"+project_name
        if hasattr(args,'time_reverse_flag') and args.time_reverse_flag !="only_forward":project_name = f"{args.time_reverse_flag}_"+project_name
        if hasattr(args,'time_intervel') and args.time_intervel:project_name = project_name + f"_per_{args.time_intervel}_step"
        if args.patch_range != args.dataset_patch_range and args.patch_range and args.dataset_patch_range: 
            project_name = project_name + f"_P{tuple2str(args.dataset_patch_range)}_for_P{tuple2str(args.patch_range)}"
        #if hasattr(args,'cross_sample') and args.cross_sample:project_name = project_name + f"_random_dataset"
        #print(project_name)
    return model_name, datasetname,project_name

def deal_with_tuple_string(patch_size,defult=None):
    if isinstance(patch_size,str):
        if len(patch_size)>0:
            patch_size  = tuple([int(t) for t in patch_size.split(',')])
            if len(patch_size) == 1: patch_size=patch_size[0]
        else:
            patch_size = defult
    elif isinstance(patch_size,int):
        patch_size = patch_size
    elif isinstance(patch_size,list):
        patch_size = tuple(patch_size)
    elif isinstance(patch_size,tuple):
        pass
    else:
        patch_size = defult
    return patch_size

def get_ckpt_path(args):
    if args.debug:
        return Path('./debug')
    TIME_NOW  = time.strftime("%m_%d_%H_%M")+f"_{args.port}" if args.distributed else time.strftime("%m_%d_%H_%M_%S")
    if args.seed == -1:args.seed = 42;#random.randint(1, 100000)
    if args.seed == -2:args.seed = random.randint(1, 100000)
    TIME_NOW  = f"{TIME_NOW}-seed_{args.seed}"
    args.trail_name = TIME_NOW
    if not hasattr(args,'train_set'):args.train_set='large'
    args.time_step  = ts_for_mode[args.mode] if not args.time_step else args.time_step
    model_name, datasetname, project_name = get_projectname(args)
    if args.continue_train or (('fourcast' in args.mode) and (not args.do_fourcast_anyway)):
        assert args.pretrain_weight
        #args.mode = "finetune"
        SAVE_PATH = Path(os.path.dirname(args.pretrain_weight))
    else:
        SAVE_PATH = Path(f'./checkpoints/{datasetname}/{model_name}/{project_name}/{TIME_NOW}')
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
    return SAVE_PATH

def parse_default_args(args):
    if args.seed == -1:args.seed = 42
    if args.seed == -2:args.seed = random.randint(1, 100000)
    args.half_model = half_model
    args.batch_size = bs_for_mode[args.mode] if args.batch_size == -1 else args.batch_size
    args.valid_batch_size = args.batch_size*8 if args.valid_batch_size == -1 else args.valid_batch_size
    args.epochs     = ep_for_mode[args.mode] if args.epochs == -1 else args.epochs
    args.lr         = lr_for_mode[args.mode] if args.lr == -1 else args.lr
    args.time_step = ts_for_mode[args.mode] if args.time_step == -1 else args.time_step

    if not hasattr(args,'epoch_save_list'):args.epoch_save_list = [99]
    if isinstance(args.epoch_save_list,str):args.epoch_save_list = [int(p) for p in args.epoch_save_list.split(',')]
    # input size
    img_size = patch_size = x_c = y_c =  dataset_type = None

    if args.train_set is not None and args.train_set in train_set:
        img_size, patch_size, x_c, y_c, dataset_type,dataset_kargs = train_set[args.train_set]
        
        if 'Euler' in args.wrapper_model: y_c  = 15
    else:
        assert args.img_size
        assert args.patch_size
        assert args.input_channel
        assert args.output_channel
        assert args.dataset_type
        dataset_kargs={}


    dataset_kargs['root'] = args.data_root if args.data_root != "" else None
    dataset_kargs['mode']        = args.mode
    dataset_kargs['time_step']   = args.time_step
    dataset_kargs['check_data']  = True
    dataset_kargs['time_reverse_flag'] = 'only_forward' if not hasattr(args,'time_reverse_flag') else args.time_reverse_flag
    
    dataset_kargs['use_offline_data'] = args.use_offline_data

    if hasattr(args,'dataset_flag') and args.dataset_flag:dataset_kargs['dataset_flag']= args.dataset_flag
    if hasattr(args,'time_intervel'):dataset_kargs['time_intervel']= args.time_intervel
    if hasattr(args,'cross_sample'):dataset_kargs['cross_sample']= args.cross_sample
    if hasattr(args,'use_time_stamp') and args.use_time_stamp:dataset_kargs['use_time_stamp']= args.use_time_stamp
    if hasattr(args,'use_position_idx'):dataset_kargs['use_position_idx']= args.use_position_idx
    
    
    args.unique_up_sample_channel = args.unique_up_sample_channel
    

    args.dataset_type = dataset_type if not args.dataset_type else args.dataset_type
    args.dataset_type = args.dataset_type.__name__ if not isinstance(args.dataset_type,str) else args.dataset_type
    x_c        = args.input_channel = x_c if not args.input_channel else args.input_channel
    y_c        = args.output_channel= y_c if not args.output_channel else args.output_channel
    patch_size = args.patch_size = deal_with_tuple_string(args.patch_size,patch_size)
    img_size   = args.img_size   = deal_with_tuple_string(args.img_size,img_size)
    patch_range= args.patch_range= deal_with_tuple_string(args.patch_range,None)
    
    if "Patch" in args.dataset_type:
        dataset_patch_range = args.dataset_patch_range = deal_with_tuple_string(args.dataset_patch_range,None)
    else:
        dataset_patch_range = args.dataset_patch_range = None
    dataset_kargs['img_size'] = img_size
    dataset_kargs['patch_range']= dataset_patch_range if dataset_patch_range else patch_range
    dataset_kargs['debug']= args.debug
    args.dataset_kargs = dataset_kargs
    args.picked_input_property = args.picked_output_property = None
    if args.picked_inputoutput_property:
        args.picked_input_property, args.picked_output_property = args.picked_inputoutput_property.split(".")
        args.picked_input_property = deal_with_tuple_string(args.picked_input_property,None)
        args.picked_input_property = [args.picked_input_property] if isinstance(args.picked_input_property,int) else args.picked_input_property
        args.picked_output_property = deal_with_tuple_string(args.picked_output_property,None)
        args.picked_output_property= [args.picked_output_property] if isinstance(args.picked_output_property,int) else args.picked_output_property
        args.input_channel = 14*len(args.picked_input_property)
        args.output_channel= 14*len(args.picked_output_property)
    dataset_kargs['picked_input_property'] = args.picked_input_property
    dataset_kargs['picked_output_property'] = args.picked_output_property
    # model_img_size= args.img_size
    # if 'Patch' in args.wrapper_model:
    #     if '3D' in args.wrapper_model:
    #         model_img_size = tuple([5]*3)
    #     else:
    #         model_img_size = tuple([5]*2)
    model_kargs={
        "img_size": args.img_size, 
        "patch_size": args.patch_size, 
        "patch_range":args.patch_range,
        "in_chans": args.input_channel, 
        "out_chans": args.output_channel,
        "fno_blocks": args.fno_blocks,
        "embed_dim": args.embed_dim if not args.debug else 32, 
        "depth": args.model_depth if not args.debug else 1,
        "debug_mode":args.debug,
        "double_skip":args.double_skip, 
        "fno_bias": args.fno_bias, 
        "fno_softshrink": args.fno_softshrink,
        "history_length": args.history_length,
        "reduce_Field_coef":args.use_scalar_advection,
        "modes":deal_with_tuple_string(args.modes,(17,33,6)),
        "mode_select":args.mode_select,
        "physics_num":args.physics_num,
        "pred_len":args.pred_len,
        "n_heads":args.n_heads,
        "label_len":args.label_len,
        "canonical_fft":args.canonical_fft,
        "unique_up_sample_channel":args.unique_up_sample_channel,
        "share_memory":args.share_memory,
        "dropout_rate":args.dropout_rate,
        "conv_simple":args.conv_simple
    }
    args.model_kargs = model_kargs


    args.snap_index = [[0,40,80,12], [t for t in [38,49,13,27] if t < args.output_channel]      # property  Z500 and T850 and v2m and u2m and 
                       ] 
    if args.wrapper_model == 'PatchWrapper':
        args.snap_index.append({0:[[15],[15]],1:[[13],[15]],2:[[11],[15]],3:[[ 9],[15]],4:[[ 7],[15]],5:[[ 5],[15]]})
    else:
        args.snap_index.append([[15,15,15, 7, 7, 7,23,23,23],
                                [15,31,45,15,31,45,15,31,45]])
    if args.output_channel<=13:args.snap_index=None
    args.real_batch_size = args.batch_size * args.accumulation_steps * args.ngpus_per_node 
    return args

def create_logsys(args,save_config=True):
    local_rank = args.gpu
    SAVE_PATH  = args.SAVE_PATH
    recorder_list = args.recorder_list if hasattr(args,'recorder_list') else ['tensorboard']
    logsys   = LoggingSystem(local_rank==0 or (not args.distributed),args.SAVE_PATH,seed=args.seed,
                             use_wandb=args.use_wandb,recorder_list=recorder_list,flag=args.mode,
                             disable_progress_bar=args.disable_progress_bar)
    hparam_dict={'patch_size':args.patch_size , 'lr':args.lr, 'batch_size':args.batch_size,
                                                   'model':args.model_type}
    metric_dict={'best_loss':None}
    dirname = SAVE_PATH
    dirname,name     = os.path.split(dirname)
    dirname,job_type = os.path.split(dirname)
    dirname,group    = os.path.split(dirname)
    dirname,project  = os.path.split(dirname)
    wandb_id         = None
    # if wandb_id is None:
    #     wandb_id = f"{project}-{group}-{job_type}-{name}"
    #     wandb_id = hashlib.md5(wandb_id.encode("utf-8")).hexdigest()+"the2"
    #args.wandb_id = wandb_id #if we dont assign the wandb_id, the default is None 
    #do not save the args.wandb_id in the config
    print(f"wandb id: {wandb_id}")
    _ = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict={'best_loss': None},
                                args=args,project = project,
                                entity  = "szztn951357",
                                group   = group,
                                job_type= job_type,
                                name    = name,
                                wandb_id =wandb_id
                               )
    # fix the seed for reproducibility
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # cudnn.benchmark = True
    ## already done in logsys
    if args.log_trace_times is None:
        logsys.log_trace_times   = 1 if "Patch" in args.dataset_type else 100
    else:
        logsys.log_trace_times = args.log_trace_times
    logsys.do_iter_log     = args.do_iter_log
    args.logsys = ""
    if not args.debug and save_config:
        for key, val in vars(args).items():
            if local_rank==0:print(f"{key:30s} ---> {val}")
        config_path = os.path.join(logsys.ckpt_root,'config.json')
        if not os.path.exists(config_path):
            with open(config_path,'w') as f:
                config = vars(args)
                config['wandb_id']=""
                json.dump(config,f)
    args.logsys = logsys
    
    return logsys

#########################################
############# main script ###############
#########################################

def build_model(args):
    #cudnn.enabled         = True
    cudnn.benchmark       = False # will search a best CNN realized way at beginning 
    cudnn.deterministic   = True # the key for continue training.
    logsys = args.logsys
    logsys.info(f"model args: img_size= {args.img_size}")
    logsys.info(f"model args: patch_size= {args.patch_size}")
    # ==============> Initial Model <=============
    if args.wrapper_model and 'Comb' in args.wrapper_model:
        assert args.model_type1
        assert args.model_type2
        args.model_kargs['in_chans'] = 55 
        args.model_kargs['out_chans'] = 13
        args.model_kargs['unique_up_sample_channel'] = 0
        args.model_kargs['history_length'] = 1
        backbone1 = eval(args.model_type1)(**args.model_kargs)
        args.model_kargs['in_chans'] = 68
        args.model_kargs['out_chans'] = 42
        args.model_kargs['history_length'] = 1
        backbone2 = eval(args.model_type2)(**args.model_kargs)
        args.model_kargs['in_chans'] = 55
        args.model_kargs['out_chans'] = 55
        args.model_kargs['history_length'] = 1
        model = eval(args.wrapper_model)(args,backbone1,backbone2,args.backbone1_ckpt_path,args.backbone2_ckpt_path)
    else:
        model = eval(args.model_type)(**args.model_kargs)
        if args.wrapper_model:
            model = eval(args.wrapper_model)(args,model)
    logsys.info(f"use model ==> {model.__class__.__name__}")
    local_rank=args.local_rank
    rank = args.rank
    if local_rank == 0:
        param_sum, buffer_sum, all_size = getModelSize(model)
        logsys.info(f"Rank: {args.rank}, Local_rank: {local_rank} | Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters= ("FED" in args.model_type)  ) 
    else:
        model = model.cuda()

    if args.half_model:model = model.half()

    model.train_mode=args.mode
    
    model.random_time_step_train = args.random_time_step
    model.input_noise_std = args.input_noise_std
    model.history_length=args.history_length
    model.use_amp = bool(args.use_amp)
    model.clip_grad= args.clip_grad
    model.pred_len = args.pred_len
    model.accumulation_steps = args.accumulation_steps

    
    model.mean_path_length = torch.zeros(1)
    if 'UVT' in args.wrapper_model:
        assert "55" in args.dataset_flag
    return model

def update_experiment_info(experiment_hub_path,epoch,args):
    try:
        if os.path.exists(experiment_hub_path):
            with open(experiment_hub_path,'r') as f:
                experiment_hub=json.load(f)
        else:
            experiment_hub={}
        path = str(args.SAVE_PATH)
        if path not in experiment_hub:
            experiment_hub[path] = {"id":len(experiment_hub),'epoch_tot':args.epochs,"start_time":time.strftime("%m_%d_%H_%M_%S")}
        experiment_hub[path]['epoch']  =epoch
        experiment_hub[path]['endtime']=time.strftime("%m_%d_%H_%M_%S")
        with open(experiment_hub_path,'w') as f:json.dump(experiment_hub, f)
    except:
        pass

class CenterWeightMSE(nn.Module):
    def __init__(self, center_range, boundary):
        super().__init__()
        self.boundary = boundary
        self.center_range = center_range
        center_x = (boundary-1)//2
        center_y = (boundary-1)//2
        weight   = torch.zeros(boundary,boundary)
        for i in range(boundary):
            for j in range(boundary):
                weight[i,j] =  np.sqrt((i - center_x)**2 + (j - center_y)**2)/(10.0*center_range/boundary)
        self.weight = weight.reshape(1,1,boundary,boundary)
    def forward(self, pred, real):
        if real.shape[-2:] != (self.boundary,self.boundary):
            return torch.mean((pred - real)**2)
        else:
            return torch.mean(((pred-real)*self.weight.to(pred.device))**2)

class PressureWeightMSE(nn.Module):
    def __init__(self, alpha=0.5,min_weight=0.1,level=14):
        super().__init__()
        self.alpha    = alpha
        self.min_weight = min_weight
        self.level = level
        self.weight = 1-torch.exp(-alpha*torch.arange(level))+min_weight
        self.weight = self.weight/self.weight.sum()*level
        self.weight = self.weight.reshape(1,1,level,1,1)
    def forward(self, pred, real):

        delta = (pred - real)**2
        B,P,W,H=delta.shape
        delta = delta.reshape(B,-1,self.level,W,H)*self.weight.to(pred.device)
        return delta.mean()

def build_optimizer(args,model):
    if args.opt == 'adamw':
        param_groups    = timm.optim.optim_factory.param_groups_weight_decay(model, args.weight_decay)
        optimizer       = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.opt == 'adam':
        optimizer       = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    elif args.opt == 'sgd':
        optimizer       = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    GDMod_type  = args.GDMod_type
    GDMod_lambda1 = args.GDMod_lambda1
    GDMod_lambda2 = args.GDMod_lambda2
    GDMode = {
        'NGmod_absolute': NGmod_absolute,
        'NGmod_delta_mean': NGmod_delta_mean,
        'NGmod_estimate_L2': NGmod_estimate_L2,
        'NGmod_absoluteNone': NGmod_absoluteNone,
        'NGmod_absolute_set_level':NGmod_absolute_set_level,
        'NGmod_RotationDeltaX':NGmod_RotationDeltaX,
        'NGmod_RotationDeltaE':NGmod_RotationDeltaE,
        'NGmod_RotationDeltaET':NGmod_RotationDeltaET,
        'NGmod_RotationDeltaETwo':NGmod_RotationDeltaETwo,
        'NGmod_RotationDeltaNmin':NGmod_RotationDeltaNmin,
        'NGmod_RotationDeltaESet':NGmod_RotationDeltaESet,
        'NGmod_RotationDeltaEThreeTwo':NGmod_RotationDeltaEThreeTwo,
        'NGmod_RotationDeltaXS':NGmod_RotationDeltaXS,
        'NGmod_RotationDeltaY':NGmod_RotationDeltaY,
        'NGmod_pathlength':NGmod_pathlength,

    }
    optimizer.grad_modifier = GDMode[GDMod_type](GDMod_lambda1, GDMod_lambda2,
        sample_times=args.GDMod_sample_times,
        L1_level=args.GDMod_L1_level,L2_level=args.GDMod_L2_level) if GDMod_type != 'off' else None
    if optimizer.grad_modifier is not None:
        optimizer.grad_modifier.ngmod_freq = args.ngmod_freq
        optimizer.grad_modifier.split_batch_chunk = args.split_batch_chunk
        optimizer.grad_modifier.update_mode = args.gmod_update_mode
        optimizer.grad_modifier.coef = None
        optimizer.grad_modifier.use_amp = bool(args.use_amp) #bool(args.gdamp)# we will force two amp same
        optimizer.grad_modifier.loss_wall = args.gd_loss_wall
        optimizer.grad_modifier.only_eval = args.gdeval
        optimizer.grad_modifier.gd_alpha  = args.gd_alpha
        optimizer.grad_modifier.alpha_stratagy = args.alpha_stratagy
        optimizer.grad_modifier.loss_target = args.gd_loss_target
        if args.gmod_coef:
            _, pixelnorm_std = np.load(args.gmod_coef)
            pixelnorm_std   = torch.Tensor(pixelnorm_std).reshape(1,70,32,64) #<--- should pad 
            assert not pixelnorm_std.isnan().any()
            assert not pixelnorm_std.isinf().any()
            # pixelnorm_std = torch.cat([pixelnorm_std[:,:55],
            #                torch.ones(1,1,32,64),
            #                pixelnorm_std[55:],
            #                torch.ones(1,1,32,64)],1
            #                )
            optimizer.grad_modifier.coef = pixelnorm_std
        optimizer.grad_modifier.path_length_regularize = args.path_length_regularize
        optimizer.grad_modifier.path_length_mode    = args.path_length_mode if args.path_length_regularize else None
        optimizer.grad_modifier.rotation_regularize = args.rotation_regularize
        optimizer.grad_modifier.rotation_regular_mode = args.rotation_regular_mode if args.rotation_regular_mode else None
    lr_scheduler = None
    if args.sched:
        lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.criterion == 'mse':
        criterion       = nn.MSELoss()
    elif args.criterion == 'pred_time_weighted_mse':
        print(args.dataset_patch_range)
        criterion=[ CenterWeightMSE(args.dataset_patch_range - i, args.dataset_patch_range) for i in range(args.time_step - args.history_length)]
    elif args.criterion == 'PressureWeightMSE':
        criterion       = PressureWeightMSE()

    return optimizer,lr_scheduler,criterion

def fast_set_model_epoch(model,**kargs):
    if hasattr(model,'set_epoch'):model.set_epoch(**kargs)
    if hasattr(model,'module') and hasattr(model.module,'set_epoch'):model.module.set_epoch(**kargs)
            


def main_worker(local_rank, ngpus_per_node, args,result_tensor=None,
        train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    if local_rank==0:print(f"we are at mode={args.mode}")
    ##### locate the checkpoint dir ###########
    args.gpu = args.local_rank = gpu  = local_rank
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    SAVE_PATH = get_ckpt_path(args)
    args.SAVE_PATH  = str(SAVE_PATH)
    ########## inital log ###################
    logsys = create_logsys(args)
    

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        logsys.info(f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

    model           = build_model(args)
    #param_groups    = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer,lr_scheduler,criterion = build_optimizer(args,model)
    loss_scaler     = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    logsys.info(f'use lr_scheduler:{lr_scheduler}')

    args.pretrain_weight = args.pretrain_weight.strip()
    logsys.info(f"loading weight from {args.pretrain_weight}")
    start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, optimizer, lr_scheduler, loss_scaler, path=args.pretrain_weight, 
                        only_model= ('fourcast' in args.mode) or (args.mode=='finetune' and not args.continue_train) ,loc = 'cuda:{}'.format(args.gpu))
    start_epoch = start_epoch if args.continue_train else 0

    if args.more_epoch_train:
        assert args.pretrain_weight
        print(f"detect more epoch training, we will do a copy processing for {args.pretrain_weight}")
        os.system(f'cp {args.pretrain_weight} {args.pretrain_weight}-epoch{start_epoch}')
    logsys.info("done!")


    # =======================> start training <==========================
    logsys.info(f"entering {args.mode} training in {next(model.parameters()).device}")
    now_best_path = SAVE_PATH / 'backbone.best.pt'
    latest_ckpt_p = SAVE_PATH / 'pretrain_latest.pt'
    test_dataloader = None
    train_loss=-1
    if args.mode=='fourcast':
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=train_dataset_tensor,test_record_load=train_record_load)
        run_fourcast(args, model,logsys,test_dataloader)
        return 1
    elif args.mode=='fourcast_for_snap_nodal_loss':
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=train_dataset_tensor,test_record_load=train_record_load)
        run_nodalosssnap(args, model,logsys,test_dataloader)
        return 1
    else:

        train_dataset, val_dataset, train_dataloader,val_dataloader = get_train_and_valid_dataset(args,
                       train_dataset_tensor=train_dataset_tensor,train_record_load=train_record_load,
                       valid_dataset_tensor=valid_dataset_tensor,valid_record_load=valid_record_load)
        logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
        logsys.info(f"Start training for {args.epochs} epochs")
        
        master_bar = logsys.create_master_bar(args.epochs)
        accu_list = ['valid_loss']
        metric_dict = logsys.initial_metric_dict(accu_list)
        banner = logsys.banner_initial(args.epochs, args.SAVE_PATH)
        logsys.banner_show(0, args.SAVE_PATH)
        val_loss=1.234
        train_loss = -1
        if args.tracemodel:logsys.wandb_watch(model,log_freq=100)
        for epoch in master_bar:
            if epoch < start_epoch:continue
            if (args.fourcast_during_train) and (epoch%args.fourcast_during_train == 0 and (epoch>0 or args.pretrain_weight)):
                if test_dataloader is None:
                    test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=None,test_record_load=None)# should disable at 
                origin_ckpt = logsys.ckpt_root
                new_ckpt  = os.path.join(logsys.ckpt_root,f'result_of_epoch_{epoch}')
                try:# in multi process will conflict
                    if new_ckpt and not os.path.exists(new_ckpt):os.makedirs(new_ckpt)
                except:
                    pass
                logsys.ckpt_root = new_ckpt
                use_amp = model.use_amp
                model.use_amp= True
                run_fourcast(args, model,logsys,test_dataloader)
                model.use_amp=use_amp 
                logsys.ckpt_root = origin_ckpt
            fast_set_model_epoch(model,epoch=epoch,epoch_total=args.epochs,eval_mode=False)
            logsys.record('learning rate',optimizer.param_groups[0]['lr'],epoch, epoch_flag='epoch')
            train_loss = run_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler,logsys,'train')
            if (not args.more_epoch_train) and (lr_scheduler is not None):lr_scheduler.step(epoch)
            #torch.cuda.empty_cache()
            #train_loss = single_step_evaluate(train_dataloader, model, criterion,epoch,logsys,status='train') if 'small' in args.train_set else -1
            fast_set_model_epoch(model,epoch=epoch,epoch_total=args.epochs,eval_mode=True)
            if (epoch%args.valid_every_epoch == 0 and not (epoch==0 and args.skip_first_valid)) or (epoch == args.epochs - 1):
                val_loss   = run_one_epoch(epoch, start_step, model, criterion, val_dataloader, optimizer, loss_scaler,logsys,'valid')
            logsys.metric_dict.update({'valid_loss':val_loss},epoch)
            logsys.banner_show(epoch,args.SAVE_PATH,train_losses=[train_loss])
            if (not args.distributed) or (args.rank == 0 and local_rank == 0) :
                logsys.info(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}",show=False)
                logsys.record('train', train_loss, epoch, epoch_flag='epoch')
                logsys.record('valid', val_loss, epoch, epoch_flag='epoch')
                if val_loss < min_loss:
                    min_loss = val_loss
                    if epoch > args.epochs//10:
                        logsys.info(f"saving best model ....",show=False)
                        save_model(model, path=now_best_path, only_model=True)
                        logsys.info(f"done;",show=False)
                    #if last_best_path is not None:os.system(f"rm {last_best_path}")
                    #last_best_path= now_best_path
                    logsys.info(f"The best accu is {val_loss}", show=False)
                logsys.record('best_loss', min_loss, epoch, epoch_flag='epoch')
                update_experiment_info(experiment_hub_path,epoch,args)
                if ((epoch>args.save_warm_up) and (epoch%args.save_every_epoch==0)) or (epoch==args.epochs-1) or (epoch in args.epoch_save_list):
                    logsys.info(f"saving latest model ....", show=False)
                    save_model(model, epoch+1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, latest_ckpt_p)
                    logsys.info(f"done ....",show=False)
                    if epoch in args.epoch_save_list:
                        save_model(model, path=f'{latest_ckpt_p}-epoch{epoch}', only_model=True)
                        #os.system(f'cp {latest_ckpt_p} {latest_ckpt_p}-epoch{epoch}')
            

        if os.path.exists(now_best_path) and args.do_final_fourcast and not args.distributed:
            logsys.info(f"we finish training, then start test on the best checkpoint {now_best_path}")
            start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, path=now_best_path, only_model=True)
            run_fourcast(args, model,logsys)
        if result_tensor is not None and local_rank==0:
            result_tensor[local_rank] = min_loss
    logsys.close()

def main_worker222(local_rank, ngpus_per_node, args,result_tensor=None,
        train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    if local_rank==0:print(f"we are at mode={args.mode}")
    ##### locate the checkpoint dir ###########
    args.gpu = args.local_rank = gpu  = local_rank
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    SAVE_PATH = get_ckpt_path(args)
    args.SAVE_PATH  = str(SAVE_PATH)
    ########## inital log ###################
    #logsys = create_logsys(args)
    

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

    data = torch.load("debug/data.pt")
    _inputs = data["_input"]#torch.randn(1,2,5,5).cuda()
    targets = data["target"]#torch.randn(1,3,5,5).cuda()
    model = nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.ReLU(),nn.Linear(5,5))
    model.load_state_dict(data['model'])
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    for name,p in model.named_parameters():
        print(f"GPU:{local_rank}:start:{name}:{p.norm().item()}")
    device=p.device
    
    grad_modifier = NGmod_absolute(10000,0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    # =======================> start training <==========================
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.MSELoss()
    optimizer.zero_grad()
    _input = _inputs[local_rank:local_rank+1].to(device)
    target = targets[local_rank:local_rank+1].to(device)
    #with torch.cuda.amp.autocast():
    output = model(_input)
    loss = loss_fn(output, target) + grad_modifier.getL1loss(model, _input)

    loss.backward()
    #scaler.scale(loss).backward()
    
    #scaler.unscale_(optimizer)
    # with torch.cuda.amp.autocast():
    #grad_modifier.backward(model, _input, output)
    
    #nn.utils.clip_grad_norm_(model.parameters(),0.001)

    for name,p in model.named_parameters():
        print(f"GPU:{local_rank}:grad:{name}:{p.grad.norm().item()}")
    optimizer.step()
    #scaler.step(optimizer)
    #scaler.update()
    for name,p in model.named_parameters():
        print(f"GPU:{local_rank}:end:{name}:{p.norm().item()}")


def create_memory_templete(args):
    train_dataset_tensor=valid_dataset_tensor=train_record_load=valid_record_load=None
    if args.use_inmemory_dataset:
        assert args.dataset_type
        print("======== loading data as shared memory==========")
        if not ('fourcast' in args.mode):
            print(f"create training dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='train' if not args.debug else 'test',
                            root=args.data_root,use_offline_data=args.use_offline_data,dataset_flag=args.dataset_flag)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load  = train_record_load.share_memory_()
            print(f"done! -> train template shape={train_dataset_tensor.shape}")
            
            print(f"create validing dataset template, .....")
            valid_dataset_tensor, valid_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='valid' if not args.debug else 'test',
                            root=args.data_root,use_offline_data=args.use_offline_data,dataset_flag=args.dataset_flag)
            valid_dataset_tensor = valid_dataset_tensor.share_memory_()
            valid_record_load  = valid_record_load.share_memory_()
            print(f"done! -> train template shape={valid_dataset_tensor.shape}")
        else:
            print(f"create testing dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='test',
                            root=args.data_root,use_offline_data=args.use_offline_data,dataset_flag=args.dataset_flag)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load  = train_record_load.share_memory_()
            print(f"done! -> test template shape={train_dataset_tensor.shape}")          
            valid_dataset_tensor = valid_record_load = None
        print("========      done        ==========")
    return train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load

def distributed_initial(args):
    import os
    ngpus = ngpus_per_node = torch.cuda.device_count()
    args.world_size = -1
    args.dist_file  = None
    args.rank       = 0
    args.dist_backend = "nccl"
    args.multiprocessing_distributed = ngpus>1
    args.ngpus_per_node = ngpus_per_node
    if not hasattr(args,'train_set'):args.train_set='large'
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", f"{64248+np.random.randint(10)}" )
    args.port = port
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    args.dist_url = f"tcp://{ip}:{port}"
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank       = int(os.environ["SLURM_PROCID"])
        jobid           = os.environ["SLURM_JOBID"]

        hostfile        = "dist_url." + jobid  + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            #with open(hostfile, "w") as f:f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    else:
        args.world_size = 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    return args

def main(args=None):
    
    if args is None:args = get_args()
    
    args = distributed_initial(args)
    train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load = create_memory_templete(args)
    result_tensor = torch.zeros(1).share_memory_()
    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = args.ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args,result_tensor,
                                    train_dataset_tensor,train_record_load,
                                    valid_dataset_tensor,valid_record_load))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, args.ngpus_per_node, args,result_tensor,
        train_dataset_tensor,train_record_load,valid_dataset_tensor,valid_record_load)
    return result_tensor


if __name__ == '__main__':
    main()
