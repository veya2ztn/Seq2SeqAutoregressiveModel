
from cmath import isnan
import os, sys,time,json,copy
import socket
from tkinter.messagebox import NO
sys.path.append(os.getcwd())
idx=0
sys.path = [p for p in sys.path if 'lustre' not in p]
hostname = socket.gethostname()
if hostname in ['SH-IDC1-10-140-0-184','SH-IDC1-10-140-0-185']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_CONSOLE']='off'
experiment_hub_path = "./experiment_hub.json"
force_big  = True
use_wandb  = False
accumulation_steps_global=4
save_intervel=100

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
from model.time_embeding_model import *
from model.physics_model import *
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import single_step_evaluate
import pandas as pd
import wandb
from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.loggingsystem import LoggingSystem


from cephdataset import (ERA5CephDataset,WeathBench71_H5,ERA5CephSmallDataset,WeathBench706,WeathBench7066,SpeedTestDataset,ERA5Tiny12_47_96,WeathBench71,WeathBench716)
#dataset_type = ERA5CephDataset
# dataset_type  = SpeedTestDataset
Datafetcher = DataSimfetcher
def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.
import random
import traceback

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
    '2D70NH5':((32,64), (2,2), 70, 70, WeathBench71_H5,{'dataset_flag':'2D70N'}),
    '2D706N':((32,64), (2,2), 70, 70, WeathBench706,{'dataset_flag':'2D70N'}),
    '2D716N':((32,64), (2,2), 71, 71, WeathBench716,{'dataset_flag':'2D71N'}),
    '3D70N':((14,32,64), (2,2,2), 5, 5, WeathBench71,{'dataset_flag':'3D70N'}),
}
#img_size, patch_size, x_c, y_c, dataset_type = train_set[args.train_set]


half_model = False

def compute_accu(ltmsv_pred, ltmsv_true):
    if len(ltmsv_pred.shape)==5:ltmsv_pred = ltmsv_pred.flatten(1,2)
    if len(ltmsv_true.shape)==5:ltmsv_true = ltmsv_true.flatten(1,2)
    if ltmsv_pred.shape[1] not in [32,720]: ltmsv_pred = ltmsv_pred.permute(0,2,3,1)
    if ltmsv_true.shape[1] not in [32,720]: ltmsv_true = ltmsv_true.permute(0,2,3,1)
    # ltmsv_pred --> (B, w,h, property)
    # ltmsv_true --> (B, w,h, property)
    w = ltmsv_pred.shape[1]
    latitude = torch.linspace(-np.pi/2,np.pi/2,w).to(ltmsv_pred.device)
    cos_lat  = torch.cos(latitude)
    latweight= cos_lat/cos_lat.mean()
    latweight = latweight.reshape(1, w, 1,1)
    # history_record <-- (B, w,h, property)
    fenzi = (latweight*ltmsv_pred*ltmsv_true).sum(dim=(1, 2))
    fenmu = torch.sqrt((latweight*ltmsv_pred**2).sum(dim=(1,2)) *
                       (latweight*ltmsv_true**2).sum(dim=(1, 2))
                       )
    return torch.clamp(fenzi/(fenmu+1e-10),0,10)

def compute_rmse(pred, true):
    if len(pred.shape)==5:pred = pred.flatten(1,2)
    if len(true.shape)==5:true = true.flatten(1,2)
    if pred.shape[1] not in [32,720]: pred = pred.permute(0,2,3,1)
    if true.shape[1] not in [32,720]: true = true.permute(0,2,3,1)
    # pred <-- (B,w,h,p)
    # true <-- (B,w,h,p)
    w         = pred.shape[1]
    latitude  = torch.linspace(-np.pi/2, np.pi/2, w).to(pred.device)
    cos_lat   = torch.cos(latitude)
    latweight = cos_lat/cos_lat.mean()
    latweight = latweight.reshape(1, w, 1,1)
    return  torch.clamp(torch.sqrt((latweight*(pred - true)**2).mean(dim=(1,2) )),0,1000)



def once_forward_with_timestamp(model,i,start,end,dataset,time_step_1_mode):
    if not isinstance(end[0],(list,tuple)):end = [end]
    start_timestamp= torch.stack([t[1] for t in start],1) #[B,T,4]
    end_timestamp = torch.stack([t[1] for t in end],1) #[B,T,4]    
    #print([(s[0].shape,s[1].shape) for s in start])
    # start is data list [ [[B,P,h,w],[B,4]] , [[B,P,h,w],[B,4]], [[B,P,h,w],[B,4]], ...]
    normlized_Field_list = dataset.do_normlize_data([[t[0] for t in start]])[0]  #always use normlized input
    normlized_Field    = torch.stack(normlized_Field_list,2) #(B,P,T,w,h)

    target_list = dataset.do_normlize_data([[t[0] for t in end]])[0]  #always use normlized input
    target   = torch.stack(target_list,2) #(B,P,T,w,h)
    
    out  = model(normlized_Field,start_timestamp, end_timestamp)
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
    else:
        start     = start[1:] + [ltmv_pred]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward(model,i,start,end,dataset,time_step_1_mode):
    if hasattr(dataset,'use_time_stamp') and dataset.use_time_stamp:
        return once_forward_with_timestamp(model,i,start,end,dataset,time_step_1_mode)
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
    for i in range(model.history_length,len(batch)):# i now is the target index
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,batch[i],dataset,time_step_1_mode)
        if extra_loss !=0:
            iter_info_pool[f'{status}_extra_loss_gpu{gpu}_timestep{i}'] = extra_loss.item()
        for extra_info_from_model in extra_info_from_model_list:
            for name, value in extra_info_from_model.items():
                iter_info_pool[f'valid_on_{status}_{name}_timestep{i}'] = value
        
        ltmv_pred = dataset.do_normlize_data([ltmv_pred])[0]
        
        abs_loss = criterion(ltmv_pred,target)
        iter_info_pool[f'{status}_abs_loss_gpu{gpu}_timestep{i}'] =  abs_loss.item()
        pred_step+=1
        loss += abs_loss + extra_loss
        diff += abs_loss
        if model.random_time_step_train and i >= random_run_step:
            break
    # loss = loss/(len(batch) - 1)
    # diff = diff/(len(batch) - 1)
    loss = loss/pred_step
    diff = diff/pred_step
    return loss, diff, iter_info_pool

def run_one_fourcast_iter(model, batch, idxes, fourcastresult,dataset,save_prediction_first_step=None,save_prediction_final_step=None):
    accu_series=[]
    rmse_series=[]
    extra_info = {}
    time_step_1_mode=False
    clim = model.clim.detach().cpu()
    start = batch[0:model.history_length] # start must be a list
    for i in range(model.history_length,len(batch)):# i now is the target index
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,batch[i],dataset,time_step_1_mode)
        for extra_info_from_model in extra_info_from_model_list:
            for key, val in extra_info_from_model.items():
                if i not in extra_info:extra_info[i] = {}
                if key not in extra_info[i]:extra_info[i][key] = []
                extra_info[i][key].append(val)

        ltmv_true = dataset.inv_normlize_data([target])[0].detach().cpu()
        ltmv_pred = ltmv_pred.detach().cpu()
        if len(clim.shape)!=len(ltmv_pred.shape):
            ltmv_pred = ltmv_pred.squeeze(-1)
            ltmv_true = ltmv_true.squeeze(-1) # temporary use this for timestamp input like [B, P, w,h,T]

        if save_prediction_first_step is not None and i==model.history_length:save_prediction_first_step[idxes] = ltmv_pred.detach().cpu()
        if save_prediction_final_step is not None and i==len(batch) - 1:save_prediction_final_step[idxes] = ltmv_pred.detach().cpu()

        accu_series.append(compute_accu(ltmv_pred - clim, ltmv_true - clim))
        #accu_series.append(compute_accu(ltmv_pred, ltmv_true ).detach().cpu())
        
        rmse_series.append(compute_rmse(ltmv_pred , ltmv_true ))
        #torch.cuda.empty_cache()
    
    
    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,20)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,20)


    for idx, accu,rmse in zip(idxes,accu_series,rmse_series):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'accu':accu,"rmse":rmse}
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
    if not isinstance(batch,list):
        batch = batch.half() if half_model else batch.float()
        channel_last = batch.shape[1] in [32,720] # (B, P, W, H )
        if channel_last:
            batch = batch.permute(0,3,1,2)
        return batch
    else:
        return [make_data_regular(x,half_model=half_model) for x in batch]



def fourcast_step(data_loader, model,logsys,random_repeat = 0):
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
    intervel = batches//100 + 1
    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        
        while inter_b.update_step():
            #if inter_b.now>10:break
            data_cost += time.time() - now;now = time.time()
            step        = inter_b.now
            idxes,batch = prefetcher.next()
            batch       = make_data_regular(batch,half_model)
            # first sum should be (B, P, W, H )
            fourcastresult,extra_info = run_one_fourcast_iter(model, batch, idxes, fourcastresult,data_loader.dataset,
                                         save_prediction_first_step=save_prediction_first_step,save_prediction_final_step=save_prediction_final_step)
            train_cost += time.time() - now;now = time.time()
            start = batch[0]
            for _ in range(random_repeat):
                batch[0] = start*(1 + torch.randn_like(start)*0.05)
                fourcastresult,extra_info = run_one_fourcast_iter(model, batch, idxes, fourcastresult,data_loader.dataset)
            rest_cost += time.time() - now;now = time.time()
            if (step+1) % intervel==0 or step==0:
                for idx, val_pool in extra_info.items():
                    info_pool={}
                    for key, val in val_pool.items():
                        logsys.record(f'test_{key}_each_fourcast_step', np.mean(val), idx)
                        info_pool[f'test_{key}_each_fourcast_step'] = np.mean(val)
                    info_pool['time_step'] = idx
                    if use_wandb:wandb.log(info_pool)
                outstring=(f"epoch:fourcast iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}] cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} [Rest]:{rest_cost/intervel:.1e}")
                inter_b.lwrite(outstring, end="\r")
    if save_prediction_first_step is not None:torch.save(save_prediction_first_step,os.path.join(logsys.ckpt_root,'save_prediction_first_step')) 
    if save_prediction_final_step is not None:torch.save(save_prediction_final_step,os.path.join(logsys.ckpt_root,'save_prediction_final_step')) 
    
    return fourcastresult

def nan_diagnose_weight(model,loss, nan_count):
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
            print(f"the value is nan in weight:{bad_weight_name}")
            raise
        else:
            nan_count+=1
            if nan_count>10:
                print("too many nan happened")
                raise
            print(f"detect nan, now at {nan_count}/10 warning level, pass....")   
            skip = True
    return loss, nan_count, skip

def nan_diagnose_grad(model,nan_count):
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
        print(f"the value is nan in weight.grad:{bad_weight_name}")
        nan_count+=1
        if nan_count>10:
            print("too many nan happened")
            raise
        print(f"detect nan, now at {nan_count}/10 warning level, pass....")   
        skip = True
    return nan_count, skip

def run_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):

    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    accumulation_steps = accumulation_steps_global # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16
    data_cost  = []
    train_cost = []
    rest_cost = []

    Fethcher   = Datafetcher
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    #raise
    batches    = len(data_loader)

    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    intervel = batches//100 + 1

    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    now = time.time()
    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        batch = prefetcher.next()
        #print(batch[0].shape)
        #raise
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now);now = time.time()
        if status == 'train':
            if hasattr(model,'set_step'):model.set_step(step=step,epoch=epoch)
            if hasattr(model,'module') and hasattr(model.module,'set_step'):model.module.set_step(step=step,epoch=epoch)
            if model.train_mode =='pretrain':
                time_truncate = max(min(epoch//3,data_loader.dataset.time_step),2)
                batch=batch[:model.history_length -1 + time_truncate]
            # the normal initial method will cause numerial explore by using timestep > 4 senenrio.
            if model.use_amp:
                with torch.cuda.amp.autocast():
                    loss, abs_loss, iter_info_pool =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            else:
                loss, abs_loss, iter_info_pool =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            loss, nan_count, skip = nan_diagnose_weight(model,loss,nan_count)
            if skip:continue
            loss /= accumulation_steps
            iter_info_pool[f'train_loss_gpu{gpu}'] =  loss.item()

            if model.use_amp:
                loss_scaler.scale(loss).backward()
            else:
                loss.backward()
            nan_count, skip = nan_diagnose_grad(model,nan_count)
            if skip:
                optimizer.zero_grad()
                continue
            if model.clip_grad:nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)
            if (step+1) % accumulation_steps == 0:
                if model.use_amp:
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                loss, abs_loss, iter_info_pool =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0 or step<30:
            iter_info_pool['iter'] = epoch*batches + step
            if use_wandb:wandb.log(iter_info_pool)
            for key, val in iter_info_pool.items():logsys.record(key, val, epoch*batches + step)
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TimeLeng]:{time_step_now:} GPU:[{gpu}] abs_loss:{abs_loss.item():.2f} loss:{loss.item():.2f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
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


def get_train_and_valid_dataset(args,train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    dataset_type   = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    train_dataset  = dataset_type(split="train" if not args.debug else 'test',dataset_tensor=train_dataset_tensor,record_load_tensor=train_record_load,**args.dataset_kargs)
    val_dataset   = dataset_type(split="valid" if not args.debug else 'test',dataset_tensor=valid_dataset_tensor,record_load_tensor=valid_record_load,**args.dataset_kargs)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None
    train_dataloader  = DataLoader(train_dataset, args.batch_size,   sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)
    val_dataloader    = DataLoader(val_dataset  , args.valid_batch_size, sampler=val_datasampler,   num_workers=8, pin_memory=True, drop_last=False)
    return   train_dataset,   val_dataset, train_dataloader, val_dataloader

def get_test_dataset(args,test_dataset_tensor=None,test_record_load=None):
    time_step = max(3*24//6,args.time_step) if args.mode=='fourcast' else 3*24//6 + args.time_step
    dataset_kargs = copy.deepcopy(args.dataset_kargs)
    dataset_kargs['time_step'] = time_step
    if dataset_kargs['time_reverse_flag'] in ['only_forward','random_forward_backward']:
        dataset_kargs['time_reverse_flag'] = 'only_forward'
    dataset_type = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    test_dataset = dataset_type(split="test", with_idx=True,dataset_tensor=test_dataset_tensor,record_load_tensor=test_record_load,**dataset_kargs)
    assert hasattr(test_dataset,'clim_tensor')
    test_datasampler  = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
    test_dataloader   = DataLoader(test_dataset, args.batch_size, sampler=test_datasampler, num_workers=8, pin_memory=False)
    return   test_dataset,   test_dataloader

def create_fourcast_metric_table(fourcastresult, logsys,test_dataset):
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
                fourcastresult[key] = val
        
    accu_list = torch.stack([p['accu'] for p in fourcastresult.values()]).mean(0)# (fourcast_num,property_num)
    accu_table= pd.DataFrame(accu_list.transpose(1,0),index=test_dataset.vnames)
    logsys.info("===>accu_table<===")
    logsys.info(accu_table);
    accu_table.to_csv(os.path.join(logsys.ckpt_root,prefix+'accu_table'))

    rmse_list = torch.stack([p['rmse'] for p in fourcastresult.values()]).mean(0)# (fourcast_num,property_num)
    rmse_table= pd.DataFrame(rmse_list.transpose(1,0),index=test_dataset.vnames)
    logsys.info("===>rmse_table<===")
    logsys.info(rmse_table);
    rmse_table.to_csv(os.path.join(logsys.ckpt_root,prefix+'rmse_table'))
    try:
        if not isinstance(test_dataset.unit_list,int):
            unit_list = torch.Tensor(test_dataset.unit_list).to(rmse_list.device)
            #print(unit_list)
            unit_num  = max(unit_list.shape)
            unit_list = unit_list.reshape(1,unit_num)
            property_num = len(test_dataset.vnames)
            if property_num > unit_num:
                assert property_num%unit_num == 0
                unit_list = torch.repeat_interleave(unit_list,int(property_num//unit_num),dim=1)
        else:
            logsys.info(f"unit list is int, ")
            unit_list= test_dataset.unit_list
        
        rmse_unit_list= (rmse_list*unit_list)
        print(rmse_unit_list.shape)
        rmse_table_unit = pd.DataFrame(rmse_unit_list.transpose(1,0),index=test_dataset.vnames)
        logsys.info("===>rmse_unit_table<===")
        logsys.info(rmse_table_unit);rmse_table_unit.to_csv(os.path.join(logsys.ckpt_root,prefix+'rmse_table_unit'))
    except:
        logsys.info(f"get wrong when use unit list, we will fource let [rmse_unit_list] = [rmse_list]")
        traceback.print_exc()

    info_pool_list = []
    for predict_time in range(len(accu_list)):
        
        real_time = (predict_time+1)*test_dataset.time_intervel*test_dataset.time_unit
        info_pool={}
        accu_table = accu_list[predict_time]
        rmse_table = rmse_list[predict_time]
        rmse_table_unit = rmse_unit_list[predict_time]
        for name,accu,rmse,rmse_unit in zip(test_dataset.vnames,accu_table,rmse_table, rmse_table_unit):
            info_pool[prefix + f'test_accu_{name}'] = accu.item()
            info_pool[prefix + f'test_rmse_{name}'] = rmse.item()
            info_pool[prefix + f'test_rmse_unit_{name}'] = rmse_unit.item()
        info_pool['real_time'] = real_time
        for key, val in info_pool.items():
            logsys.record(key,val, predict_time)
        info_pool['time_step'] = predict_time
        if use_wandb:wandb.log(info_pool)
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
    save_path = os.path.join(logsys.ckpt_root,f"fourcastresult.gpu_{gpu}")
    if not os.path.exists(save_path) or  args.force_fourcast:
        logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
        logsys.info("starting fourcast~!")
        with open(os.path.join(logsys.ckpt_root,'weight_path'),'w') as f:f.write(args.pretrain_weight)
        fourcastresult = fourcast_step(test_dataloader, model,logsys,random_repeat = args.fourcast_randn_initial)
        torch.save(fourcastresult,save_path)
        logsys.info(f"save fourcastresult at {save_path}")
    else:
        logsys.info(f"load fourcastresult at {save_path}")
        fourcastresult = torch.load(save_path)

    if not args.distributed:
        create_fourcast_metric_table(fourcastresult, logsys,test_dataset)
    
    return 1




def get_model_name(args):
    model_name = args.model_type
    if "FED" in args.model_type:
        mode_name =args.modes.replace(",","-")
        return f"{args.model_type}.{args.mode_select}_select.M{mode_name}_P{args.pred_len}L{args.label_len}"
    if "AFN" in args.model_type and hasattr(args,'model_depth') and args.model_depth == 6:
        model_name = "small_" + model_name
    model_name = f"ViT_in_bulk-{model_name}" if len(args.img_size)>2 else model_name
    model_name = f"{args.wrapper_model}-{model_name}" if args.wrapper_model else model_name
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

    project_name = f"{args.mode}-{args.train_set}"
    if hasattr(args,'random_time_step') and args.random_time_step:project_name = 'random_step_'+project_name 
    if hasattr(args,'time_step') and args.time_step:              project_name = f"time_step_{args.time_step}_" +project_name 
    if hasattr(args,'history_length') and args.history_length !=1:project_name = f"history_{args.history_length}_"+project_name
    if hasattr(args,'time_reverse_flag') and args.time_reverse_flag !="only_forward":project_name = f"{args.time_reverse_flag}_"+project_name
    if hasattr(args,'time_intervel') and args.time_intervel:project_name = project_name + f"_every_{args.time_intervel}_step"
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
    if args.debug: return Path('./debug')
    TIME_NOW  = time.strftime("%m_%d_%H_%M") if args.distributed else time.strftime("%m_%d_%H_%M_%S")
    if args.seed == -1:args.seed = 42;#random.randint(1, 100000)
    TIME_NOW  = f"{TIME_NOW}-seed_{args.seed}"
    args.trail_name = TIME_NOW
    if not hasattr(args,'train_set'):args.train_set='large'
    args.time_step  = ts_for_mode[args.mode] if not args.time_step else args.time_step
    model_name, datasetname, project_name = get_projectname(args)
    if (args.pretrain_weight and args.mode!='finetune') or args.continue_train:
        #args.mode = "finetune"
        SAVE_PATH = Path(os.path.dirname(args.pretrain_weight))
    else:
        SAVE_PATH = Path(f'./checkpoints/{datasetname}/{model_name}/{project_name}/{TIME_NOW}')
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
    return SAVE_PATH

def parse_default_args(args):
    if args.seed == -1:args.seed = 42
    args.half_model = half_model
    args.accumulation_steps_global=accumulation_steps_global
    args.batch_size = bs_for_mode[args.mode] if args.batch_size == -1 else args.batch_size
    args.valid_batch_size = args.batch_size*8 if args.valid_batch_size == -1 else args.valid_batch_size
    args.epochs     = ep_for_mode[args.mode] if args.epochs == -1 else args.epochs
    args.lr         = lr_for_mode[args.mode] if args.lr == -1 else args.lr
    args.time_step = ts_for_mode[args.mode] if args.time_step == -1 else args.time_step
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
    if hasattr(args,'dataset_flag') and args.dataset_flag:dataset_kargs['dataset_flag']= args.dataset_flag
    if hasattr(args,'time_intervel'):dataset_kargs['time_intervel']= args.time_intervel
    if hasattr(args,'use_time_stamp') and args.use_time_stamp:dataset_kargs['use_time_stamp']= args.use_time_stamp
    
    

    args.dataset_type = dataset_type if not args.dataset_type else args.dataset_type
    args.dataset_type = args.dataset_type.__name__ if not isinstance(args.dataset_type,str) else args.dataset_type
    x_c        = args.input_channel = x_c if not args.input_channel else args.input_channel
    y_c        = args.output_channel= y_c if not args.output_channel else args.output_channel
    patch_size = args.patch_size = deal_with_tuple_string(args.patch_size,patch_size)
    img_size   = args.img_size   = deal_with_tuple_string(args.img_size,img_size)
    
    args.dataset_kargs = dataset_kargs
    
    model_kargs={
        "img_size": args.img_size, 
        "patch_size": args.patch_size, 
        "in_chans": args.input_channel, 
        "out_chans": args.output_channel,
        "fno_blocks": args.fno_blocks,
        "embed_dim": args.embed_dim if not args.debug else 16, 
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
        "label_len":args.label_len,
    }
    args.model_kargs = model_kargs
    return args

def create_logsys(args,save_config=True):
    local_rank = args.gpu
    SAVE_PATH  = args.SAVE_PATH
    if local_rank     == 0:
        dirname = SAVE_PATH
        dirname,name     = os.path.split(dirname)
        dirname,job_type = os.path.split(dirname)
        dirname,group    = os.path.split(dirname)
        dirname,project  = os.path.split(dirname)
        if use_wandb:wandb.init(config  = args,
            project = project,
            entity  = "szztn951357",
            group   = group,
            job_type= job_type,
            name    = name,
            #dir     = "ERA5_20-12/AFNONet/pretrain-physics_small/08_11_21_34",
            settings=wandb.Settings(_disable_stats=True)
            #reinit=True
            )
    logsys   = LoggingSystem(local_rank==0 or not args.distributed,args.SAVE_PATH,seed=args.seed)
    _        = logsys.create_recorder(hparam_dict={'patch_size':args.patch_size , 'lr':args.lr, 'batch_size':args.batch_size,
                                                   'model':args.model_type},
                                      metric_dict={'best_loss':None})
    # fix the seed for reproducibility
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # cudnn.benchmark = True
    ## already done in logsys
    
    if not args.debug and save_config:
        for key, val in vars(args).items():
            if local_rank==0:print(f"{key:30s} ---> {val}")
        config_path = os.path.join(logsys.ckpt_root,'config.json')
        if not os.path.exists(config_path):
            with open(config_path,'w') as f:
                json.dump(vars(args),f)
    args.logsys = logsys
    return logsys

def build_model(args):
    logsys = args.logsys
    logsys.info(f"model args: img_size= {args.img_size}")
    logsys.info(f"model args: patch_size= {args.patch_size}")
    # ==============> Initial Model <=============
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = model.cuda()

    if args.half_model:model = model.half()

    model.train_mode=args.mode
    model.random_time_step_train = args.random_time_step
    model.input_noise_std = args.input_noise_std
    model.history_length=args.history_length
    model.use_amp = args.use_amp
    model.clip_grad= args.clip_grad
    return model

def update_experiment_info(experiment_hub_path,epoch,args):
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


def main_worker(local_rank, ngpus_per_node, args,
        train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    print(f"we are at mode={args.mode}")
    ##### locate the checkpoint dir ###########
    args.gpu = args.local_rank = gpu  = local_rank
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    SAVE_PATH = get_ckpt_path(args)
    args.SAVE_PATH = str(SAVE_PATH)
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
    param_groups    = timm.optim.optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer       = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler     = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion       = nn.MSELoss()


    logsys.info(f"loading weight from {args.pretrain_weight}")
    start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, optimizer, lr_scheduler, loss_scaler, path=args.pretrain_weight, 
                        only_model= (args.mode=='fourcast') or (args.mode=='finetune' and not args.continue_train) ,loc = 'cuda:{}'.format(args.gpu))
    logsys.info("done!")


    # =======================> start training <==========================
    print(f"entering {args.mode} training in {next(model.parameters()).device}")
    now_best_path = SAVE_PATH / 'backbone.best.pt'
    latest_ckpt_p = SAVE_PATH / 'pretrain_latest.pt'

    if args.mode=='fourcast':
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=train_dataset_tensor,
                                      test_record_load=train_record_load)
        run_fourcast(args, model,logsys,test_dataloader)
        return 1
    else:
        train_dataset, val_dataset, train_dataloader,val_dataloader = get_train_and_valid_dataset(args,
                       train_dataset_tensor=train_dataset_tensor,train_record_load=train_record_load,
                       valid_dataset_tensor=valid_dataset_tensor,valid_record_load=valid_record_load)
        logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
        logsys.info(f"Start training for {args.epochs} epochs")
        master_bar        = logsys.create_master_bar(args.epochs)
        for epoch in master_bar:
            if epoch < start_epoch:continue
            if hasattr(model,'set_epoch'):model.set_epoch(epoch=epoch,epoch_total=args.epochs)
            if hasattr(model,'module') and hasattr(model.module,'set_epoch'):model.module.set_epoch(epoch=epoch,epoch_total=args.epochs)
            train_loss = run_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler,logsys,'train')
            lr_scheduler.step(epoch)
            #torch.cuda.empty_cache()
            #train_loss = single_step_evaluate(train_dataloader, model, criterion,epoch,logsys,status='train') if 'small' in args.train_set else -1
            val_loss   = run_one_epoch(epoch, start_step, model, criterion, val_dataloader, optimizer, loss_scaler,logsys,'valid')

            if (not args.distributed) or (args.rank == 0 and local_rank == 0) :
                logsys.info(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                logsys.record('train', train_loss, epoch)
                logsys.record('valid', val_loss, epoch)
                if use_wandb:wandb.log({"epoch":epoch,'train':train_loss,'valid':val_loss})
                if val_loss < min_loss:
                    min_loss = val_loss
                    if epoch > args.epochs//10:
                        logsys.info(f"saving best model ....")
                        save_model(model, path=now_best_path, only_model=True)
                        logsys.info(f"done;")
                    #if last_best_path is not None:os.system(f"rm {last_best_path}")
                    #last_best_path= now_best_path
                    logsys.info(f"The best accu is {val_loss}")
                logsys.record('best_loss', min_loss, epoch)
                if epoch>args.save_warm_up:
                    logsys.info(f"saving latest model ....")
                    save_model(model, epoch+1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, latest_ckpt_p)
                    update_experiment_info(experiment_hub_path,epoch,args)
                    logsys.info(f"done ....")
        
        if os.path.exists(now_best_path) and args.do_final_fourcast:
            logsys.info(f"we finish training, then start test on the best checkpoint {now_best_path}")
            start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, path=now_best_path, only_model=True)
            run_fourcast(args, model,logsys)
        if use_wandb:wandb.finish()
        return {'valid_loss':min_loss}


def main(args=None):
    import os
    if args is None:
        args = get_args()
    ngpus = ngpus_per_node = torch.cuda.device_count()
    args.world_size = -1
    args.dist_file  = None
    args.rank       = 0
    args.dist_backend = "nccl"
    args.multiprocessing_distributed = ngpus>1
    if not hasattr(args,'train_set'):args.train_set='large'
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
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
    
    train_dataset_tensor=valid_dataset_tensor=train_record_load=valid_record_load=None
    if args.use_inmemory_dataset:
        assert args.dataset_type
        print("======== loading data as shared memory==========")
        if not args.mode=='fourcast':
            print(f"create training dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='train' if not args.debug else 'test',root=args.data_root)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load  = train_record_load.share_memory_()
            print(f"done! -> train template shape={train_dataset_tensor.shape}")
            
            print(f"create validing dataset template, .....")
            valid_dataset_tensor, valid_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='valid' if not args.debug else 'test',root=args.data_root)
            valid_dataset_tensor = valid_dataset_tensor.share_memory_()
            valid_record_load  = valid_record_load.share_memory_()
            print(f"done! -> train template shape={valid_dataset_tensor.shape}")
        else:
            print(f"create testing dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='test',root=args.data_root)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load  = train_record_load.share_memory_()
            print(f"done! -> test template shape={train_dataset_tensor.shape}")          
            valid_dataset_tensor = valid_record_load = None
        print("========      done        ==========")

    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,
                                    train_dataset_tensor,train_record_load,
                                    valid_dataset_tensor,valid_record_load))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, ngpus_per_node, args,train_dataset_tensor,train_record_load,
                            valid_dataset_tensor,valid_record_load)

if __name__ == '__main__':
    main()
