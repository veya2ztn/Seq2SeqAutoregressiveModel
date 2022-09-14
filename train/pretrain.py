
from cmath import isnan
import os, sys,time,json
sys.path.append(os.getcwd())
idx=0
sys.path = [p for p in sys.path if 'lustre' not in p]
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_MODE'] = 'offline'
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
from model.physics_model import *
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import single_step_evaluate
import pandas as pd
import wandb
from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.loggingsystem import LoggingSystem


from cephdataset import (ERA5CephDataset,ERA5CephSmallDataset,SpeedTestDataset,load_test_dataset_in_memory,
                         load_small_dataset_in_memory,ERA5CephInMemoryDataset,mean_std_ERA5_20,\
                         ERA5Tiny12_47_96,ERA5Tiny12_47_96_Normal)
#dataset_type = ERA5CephDataset
# dataset_type  = SpeedTestDataset

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.
import random


lr_for_mode={
    'pretrain':5e-4,
    'finetune':1e-4,
    'fourcast':1e-4
}
ep_for_mode={
    'pretrain':80,
    'finetune':50,
    'fourcast':50
}
bs_for_mode={
    'pretrain':4,
    'finetune':3,
    'fourcast':3
}
as_for_mode={
    'pretrain':8,
    'finetune':16,
    'fourcast':16
}
ts_for_mode={'pretrain':2,
             'finetune':3,'fourcast':36}

train_set={
    'large': ((720, 1440), 8, 20, 20, ERA5CephDataset),
    'small': ( (32,   64), 8, 20, 20, ERA5CephSmallDataset),
    'test_large': ((720, 1440), 8, 20, 20, lambda **kargs:SpeedTestDataset(720,1440,**kargs)),
    'test_small': ( (32,   64), 8, 20, 20, lambda **kargs:SpeedTestDataset( 32,  64,**kargs)),
    'physics_small': ( (32,   64), 2, 12, 12, ERA5CephSmallDataset),
    'physics': ((720, 1440), 8, 12, 12, ERA5CephDataset),
}



half_model = False
last_best_path=None

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

def run_one_iter(model, batch, criterion, status, gpu, dataset):
    iter_info_pool={}
    loss = 0
    diff = 0
    time_step_1_mode=False
    random_run_step = np.random.randint(1,len(batch)) if len(batch)>1 else 0
    if len(batch) == 1 and isinstance(batch[0],(list,tuple)) and len(batch[0])>1:
        batch = batch[0] # (Field, FieldDt)
        time_step_1_mode=True
    with torch.cuda.amp.autocast():
        start = batch[0]
        
        for i in range(1,len(batch)):
            if isinstance(start,(list,tuple)):# now is [Field, Field_Dt, physics_part]
                Field  = start[0]

                normlized_Field, _ , Advection    = dataset.do_normlize_data([start])[0]
                target = dataset.do_normlize_data([batch[i][0]])[0] # in standand unit
            elif time_step_1_mode:
                normlized_Field, target = dataset.do_normlize_data([[start,batch[1]]])[0]
            else:
                normlized_Field = dataset.do_normlize_data([start])[0]  #always use normlized input
                target          = dataset.do_normlize_data([batch[i]])[0] #always use normlized target

            if model.training and model.input_noise_std and i==1:
                normlized_Field += torch.randn_like(normlized_Field)*model.input_noise_std
  
            out   = model(normlized_Field)

            extra_loss = 0
            if isinstance(out,(list,tuple)):
                extra_loss=out[1]
                iter_info_pool[f'{status}_extra_loss_gpu{gpu}_timestep{i}'] = extra_loss.item()
                for extra_info_from_model in out[2:]:
                    for name, value in extra_info_from_model.items():
                        iter_info_pool[f'valid_on_{status}_{name}_timestep{i}'] = value
                out = out[0]

            if isinstance(batch[0],(list,tuple)):
                normlized_Deltat_F  = out
                _, Deltat_F = dataset.inv_normlize_data([[0,normlized_Deltat_F]])[0]
                ltmv_pred   = Field + Deltat_F - Advection
                start = [ltmv_pred, 0 , batch[i][-1]]
            else:
                ltmv_pred = start = dataset.inv_normlize_data([out])[0] 

            abs_loss = criterion(dataset.do_normlize_data([ltmv_pred])[0],target)
            iter_info_pool[f'{status}_abs_loss_gpu{gpu}_timestep{i}'] =  abs_loss.item()
            loss += (abs_loss + extra_loss)/(len(batch) - 1)
            diff += abs_loss/(len(batch) - 1)
            if model.random_time_step_train and i >= random_run_step:
                break
    return loss, diff, iter_info_pool

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

def run_one_fourcast_iter(model, batch, idxes, fourcastresult,dataset):
    accu_series=[]
    rmse_series=[]
    start = batch[0]
    extra_info = {}
    clim = model.clim
    for i in range(1,len(batch)):
        if isinstance(start,(list,tuple)):# now is [Field, Field_Dt, physics_part]
            Field  = start[0]
            normlized_Field, _ , Advection = dataset.do_normlize_data([start])[0]
            target = batch[i][0] # in standand unit
        else:
            normlized_Field = dataset.do_normlize_data([start])[0]
            target = batch[i]
        out   = model(normlized_Field)
        extra_loss = 0
        if isinstance(out,(list,tuple)):
            extra_loss=out[1]
            for extra_info_from_model in out[2:]:
                for key, val in extra_info_from_model.items():
                    if i not in extra_info:extra_info[i] = {}
                    if key not in extra_info[i]:extra_info[i][key] = []
                    extra_info[i][key].append(val)
            out = out[0]
        

        if isinstance(batch[0],(list,tuple)):
            normlized_Deltat_F  = out
            _, Deltat_F = dataset.inv_normlize_data([[0,normlized_Deltat_F]])[0]
            ltmv_pred = Field + Deltat_F - Advection
            start = [ltmv_pred, 0 , batch[i][-1]]
        else:
            out = dataset.inv_normlize_data([out])[0] 
            ltmv_pred = start = out
        ltmv_true = target# (B, P, W, H )
        
        accu_series.append(compute_accu(ltmv_pred - clim, ltmv_true - clim).detach().cpu())
        #accu_series.append(compute_accu(ltmv_pred, ltmv_true ).detach().cpu())
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
        if channel_last:batch = batch.permute(0,3,1,2) 
        return batch
    else:
        return [make_data_regular(x,half_model=half_model) for x in batch]



def fourcast_step(data_loader, model,logsys,random_repeat = 0):
    model.eval()
    logsys.eval()
    status     = 'test'
    gpu        = dist.get_rank() if hasattr(model,'module') else 0
    Fethcher   = DataSimfetcher
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
            #if inter_b.now>10:break
            data_cost += time.time() - now;now = time.time()
            step        = inter_b.now
            idxes,batch = prefetcher.next()

            batch       = make_data_regular(batch,half_model)
            # first sum should be (B, P, W, H )
            fourcastresult,extra_info = run_one_fourcast_iter(model, batch, idxes, fourcastresult,data_loader.dataset)
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
    return fourcastresult


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
    data_cost = train_cost = rest_cost = 0
    Fethcher   = DataSimfetcher
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    batches    = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    intervel = batches//100 + 1

    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    now = time.time()
    total_diff,total_num  = 0,0

    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        batch = prefetcher.next()
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost += time.time() - now;now = time.time()
        if status == 'train':
            if model.train_mode =='pretrain':
                time_truncate = max(min(epoch//3,data_loader.dataset.time_step),2)
                batch=batch[:time_truncate]
            # the normal initial method will cause numerial explore by using timestep > 4 senenrio.
            loss, abs_loss, iter_info_pool =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            
            if torch.isnan(loss):raise
            loss /= accumulation_steps
            iter_info_pool[f'train_loss_gpu{gpu}'] =  loss.item()
            loss_scaler.scale(loss).backward()
            # 梯度累积
            if (step+1) % accumulation_steps == 0:
            #if half_model:
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                loss, abs_loss, iter_info_pool =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
        total_diff += abs_loss.item()
        total_num  += len(batch) - 1

        train_cost += time.time() - now;now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0 or step==1:
            iter_info_pool['iter'] = epoch*batches + step
            if use_wandb:wandb.log(iter_info_pool)
            for key, val in iter_info_pool.items():logsys.record(key, val, epoch*batches + step)
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TimeLeng]:{time_step_now:} GPU:[{gpu}] abs_loss:{abs_loss.item():.2f} loss:{loss.item():.2f} cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} ")
            data_cost = train_cost = rest_cost = 0
            inter_b.lwrite(outstring, end="\r")


    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    loss_val = total_diff/ total_num
    return loss_val


def get_train_and_valid_dataset(args,train_dataset_tensor=None,valid_dataset_tensor=None):
    train_dataset = args.dataset_type(split="train", mode=args.mode, time_step=args.time_step, check_data=True,dataset_tensor=train_dataset_tensor,
                                enable_physics_dataset=args.activate_physics_dataset)
    val_dataset   = args.dataset_type(split="valid", mode=args.mode, time_step=args.time_step, check_data=True,dataset_tensor=valid_dataset_tensor,
                                enable_physics_dataset=args.activate_physics_dataset)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None
    train_dataloader  = DataLoader(train_dataset, args.batch_size,   sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)
    val_dataloader    = DataLoader(val_dataset  , args.batch_size*8, sampler=val_datasampler,   num_workers=8, pin_memory=True, drop_last=False)
    return   train_dataset,   val_dataset, train_dataloader, val_dataloader

def get_test_dataset(args,train_dataset_tensor=None):
    time_step = max(3*24//6,args.time_step) if args.mode=='fourcast' else 3*24//6 + args.time_step
    # if args.dataset_type.__class__.__name__=='ERA5CephSmallDataset':
    #     dataset_type  = ERA5CephInMemoryDataset
    #     test_dataset  = dataset_type(split="test" , mode=args.mode,years=[2018],root="datasets/era5G32x64_new",
    #                                     check_data=True,
    #                                     dataset_tensor=train_dataset_tensor,
    #                                     with_idx=True,
    #                                     time_step=time_step,enable_physics_dataset=args.activate_physics_dataset)
    #     print("A")
    # else:
    test_dataset = args.dataset_type(split="test", mode=args.mode, check_data=True,dataset_tensor=train_dataset_tensor,
                                    time_step=time_step,with_idx=True,enable_physics_dataset=args.activate_physics_dataset)

    assert hasattr(test_dataset,'clim_tensor')
    test_datasampler  = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
    test_dataloader   = DataLoader(test_dataset, args.batch_size, sampler=test_datasampler, num_workers=8, pin_memory=False)
    return   test_dataset,   test_dataloader

def run_fourcast(args, model,logsys):
    import warnings
    warnings.filterwarnings("ignore")

    test_dataset,  test_dataloader = get_test_dataset(args)
    logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
    logsys.info("starting fourcast~!")
    with open(os.path.join(logsys.ckpt_root,'weight_path'),'w') as f:f.write(args.pretrain_weight)
    fourcastresult = fourcast_step(test_dataloader, model,logsys,random_repeat = args.fourcast_randn_initial)
    gpu       = dist.get_rank() if hasattr(model,'module') else 0
    save_path = os.path.join(logsys.ckpt_root,f"fourcastresult.gpu_{gpu}")
    torch.save(fourcastresult,save_path)
    logsys.info(f"save fourcastresult at {save_path}")

    if not args.distributed:

        accu_list = torch.stack([p['accu'] for p in fourcastresult.values()]).mean(0)# (fourcast_num,property_num)
        accu_table= pd.DataFrame(accu_list.transpose(1,0),index=test_dataset.vnames)
        logsys.info("===>accu_table<===")
        logsys.info(accu_table);
        accu_table.to_csv(os.path.join(logsys.ckpt_root,'accu_table'))

        rmse_list = torch.stack([p['rmse'] for p in fourcastresult.values()]).mean(0)# (fourcast_num,property_num)
        rmse_table= pd.DataFrame(rmse_list.transpose(1,0),index=test_dataset.vnames)
        logsys.info("===>rmse_table<===")
        logsys.info(rmse_table);
        rmse_table.to_csv(os.path.join(logsys.ckpt_root,'rmse_table'))

        if args.dataset_type in ["",'ERA5CephDataset','ERA5CephSmallDataset']:
            unit_list = torch.Tensor([mean_std_ERA5_20[name]['std'] for name in test_dataset.vnames]).to(rmse_list.device)
            unit_list = unit_list.reshape(1,len(test_dataset.vnames))# (1,property_num)
            rmse_unit_list= (rmse_list*unit_list)
            rmse_table_unit = pd.DataFrame(rmse_unit_list.transpose(1,0),index=test_dataset.vnames)
            logsys.info("===>rmse_unit_table<===")
            logsys.info(rmse_table_unit);rmse_table_unit.to_csv(os.path.join(logsys.ckpt_root,'rmse_table_unit'))
        else:
            rmse_unit_list = rmse_list

        for predict_time in range(len(accu_list)):
            info_pool={}
            accu_table = accu_list[predict_time]
            rmse_table = rmse_list[predict_time]
            rmse_table_unit = rmse_unit_list[predict_time]
            for name,accu,rmse,rmse_unit in zip(test_dataset.vnames,accu_table,rmse_table, rmse_table_unit):
                info_pool[f'test_accu_{name}'] = accu.item()
                info_pool[f'test_rmse_{name}'] = rmse.item()
                info_pool[f'test_rmse_unit_{name}'] = rmse_unit.item()
            for key, val in info_pool.items():
                logsys.record(key,val, predict_time)
            info_pool['time_step'] = predict_time
            if use_wandb:wandb.log(info_pool)
    return 1

def get_model_name(args):
    model_name = args.model_type
    model_name = f"ViT_in_bulk-{model_name}" if len(args.img_size)>2 else model_name
    model_name = f"{args.wrapper_model}-{model_name}" if args.wrapper_model else model_name
    return model_name

def get_datasetname(args):
    if args.dataset_type in ["",'ERA5CephDataset','ERA5CephSmallDataset']:
        assert args.train_set in train_set
        datasetname  = "ERA5_20-12" if 'physics' in args.train_set else "ERA5_20"
    else:
        datasetname = args.dataset_type
    return datasetname

def get_projectname(args):
    model_name   = get_model_name(args)
    datasetname  = get_datasetname(args)

    project_name =f"{args.mode}-{args.train_set}"
    project_name = 'random_step_'+project_name if args.random_time_step else project_name
    project_name = f"time_step_{args.time_step}_"+project_name if args.time_step else project_name
    return model_name, datasetname, project_name
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

def main_worker(local_rank, ngpus_per_node, args, train_dataset_tensor=None,valid_dataset_tensor=None):
    print(f"we are at mode={args.mode}")
    TIME_NOW  = time.strftime("%m_%d_%H_%M") if args.distributed else time.strftime("%m_%d_%H_%M_%S")
    if args.seed == -1:args.seed = random.randint(1, 100000)
    TIME_NOW  = f"{TIME_NOW}-seed_{args.seed}"
    args.trail_name = TIME_NOW
    if not hasattr(args,'train_set'):
        args.train_set='large'

    args.time_step  = ts_for_mode[args.mode] if not args.time_step else args.time_step

    model_name, datasetname, project_name = get_projectname(args)

    if args.pretrain_weight and args.mode=='fourcast':
        #args.mode = "finetune"
        SAVE_PATH = Path(os.path.dirname(args.pretrain_weight))
    else:
        SAVE_PATH = Path(f'./checkpoints/{datasetname}/{model_name}/{project_name}/{TIME_NOW}')
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    ngpus = torch.cuda.device_count()

    args.gpu = gpu  = local_rank
    args.half_model = half_model
    args.accumulation_steps_global=accumulation_steps_global
    args.batch_size = bs_for_mode[args.mode] if args.batch_size == -1 else args.batch_size
    args.epochs     = ep_for_mode[args.mode] if args.epochs == -1 else args.epochs
    args.lr         = lr_for_mode[args.mode] if args.lr == -1 else args.lr
    # input size
    img_size = patch_size = x_c = y_c =  dataset_type = None
    if args.train_set is not None and args.train_set in train_set:
        img_size, patch_size, x_c, y_c, dataset_type = train_set[args.train_set]
        if 'Euler' in args.wrapper_model: y_c  = 15
        if len(img_size)>2:
            x_c, y_c = 4, 4
        args.activate_physics_dataset = 'physics' in args.train_set
    else:
        assert args.img_size
        assert args.patch_size
        assert args.input_channel
        assert args.output_channel
        assert args.dataset_type
    x_c = input_channel  = x_c if not args.input_channel else args.input_channel
    y_c = output_channel = y_c if not args.output_channel else args.output_channel
    patch_size = args.patch_size = deal_with_tuple_string(args.patch_size,patch_size)
    img_size   = args.img_size   = deal_with_tuple_string(args.img_size,img_size)


    ########## inital log ###################
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
    logsys   = LoggingSystem(local_rank==0 or not args.distributed,SAVE_PATH,seed=args.seed)
    _        = logsys.create_recorder(hparam_dict={'patch_size':patch_size , 'lr':args.lr, 'batch_size':args.batch_size,
                                                   'model':args.model_type},
                                      metric_dict={'best_loss':None})
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #cudnn.benchmark = True

    for key, val in vars(args).items():
        print(f"{key:30s} ---> {val}")
    config_path = os.path.join(logsys.ckpt_root,'config.json')
    if not os.path.exists(config_path):
        with open(config_path,'w') as f:
            json.dump(vars(args),f)

    args.dataset_type = dataset_type if not args.dataset_type else eval(args.dataset_type)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        logsys.info(f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

    print(f"model args: img_size= {img_size}")
    print(f"model args: patch_size= {patch_size}")
    # ==============> Initial Model <=============
    if args.distributed or force_big:
        model = eval(args.model_type)(img_size=img_size, patch_size=patch_size, in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        fno_blocks=args.fno_blocks,
                        double_skip=args.double_skip, fno_bias=args.fno_bias, fno_softshrink=args.fno_softshrink,
                        )
    else:
        model = eval(args.model_type)(img_size=img_size, patch_size=patch_size, in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        fno_blocks=args.fno_blocks,
                        double_skip=args.double_skip, fno_bias=args.fno_bias, fno_softshrink=args.fno_softshrink,
                        embed_dim=16, depth=1,debug_mode=1
                        )
    if args.wrapper_model:
        model = eval(args.wrapper_model)(args,model)
    #model = hfnn.to_hfai(model)
    model.train_mode=args.mode
    logsys.info(f"use model ==> {model.__class__.__name__}")
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
    if args.half_model:
        model = model.half()
    model.random_time_step_train = args.random_time_step
    model.input_noise_std = args.input_noise_std
    # args.lr = args.lr * args.batch_size * dist.get_world_size() / 512.0
    param_groups    = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer       = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler     = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion       = nn.MSELoss()

    if args.mode == 'pretrain':
        start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model,
                                        optimizer, lr_scheduler, loss_scaler,
                                        SAVE_PATH/'pretrain_latest.pt',loc = 'cuda:{}'.format(args.gpu))
    else:
        if (SAVE_PATH / 'finetune_latest.pt').exists():
            start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, optimizer, lr_scheduler, loss_scaler, SAVE_PATH / 'finetune_latest.pt')
        else:
            #assert args.pretrain_weight != ""
            #assert os.path.exists(args.pretrain_weight)
            logsys.info(f"loading weight from {args.pretrain_weight}")
            start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, path=args.pretrain_weight, only_model=True)
            logsys.info("done!")


    # =======================> start training <==========================
    print(f"entering {args.mode} training")

    if args.mode=='fourcast':
        run_fourcast(args, model,logsys)
    else:
        train_dataset, val_dataset, train_dataloader,val_dataloader = get_train_and_valid_dataset(args,
                       train_dataset_tensor=train_dataset_tensor,valid_dataset_tensor=valid_dataset_tensor)
        if local_rank == 0:
            logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
            logsys.info(f"Start training for {args.epochs} epochs")

        master_bar        = logsys.create_master_bar(args.epochs)
        last_best_path = None
        for epoch in master_bar:
            if epoch < start_epoch:continue
            train_loss = run_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler,logsys,'train')
            lr_scheduler.step(epoch)
            #torch.cuda.empty_cache()
            #train_loss = single_step_evaluate(train_dataloader, model, criterion,epoch,logsys,status='train') if 'small' in args.train_set else -1
            val_loss   = run_one_epoch(epoch, start_step, model, criterion, val_dataloader, optimizer, loss_scaler,logsys,'valid')

            if (not args.distributed) or (rank == 0 and local_rank == 0) :
                logsys.info(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                logsys.record('train', train_loss, epoch)
                logsys.record('valid', val_loss, epoch)
                if use_wandb:wandb.log({"epoch":epoch,'train':train_loss,'valid':val_loss})
                if val_loss < min_loss:
                    min_loss = val_loss
                    now_best_path = SAVE_PATH / f'backbone.best.pt'
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
                    save_model(model, epoch+1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')
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
    train_dataset_tensor=valid_dataset_tensor=None


    if args.dataset_type in ['ERA5CephDataset','ERA5CephSmallDataset']:
        print("======== loading data as shared memory==========")
        if args.dataset_type == 'ERA5CephSmallDataset':
            if not args.mode=='fourcast':
                train_dataset_tensor = load_small_dataset_in_memory('train').share_memory_()
                print(f"train->{train_dataset_tensor.shape}")
                valid_dataset_tensor = load_small_dataset_in_memory('valid').share_memory_()
                print(f"valid->{train_dataset_tensor.shape}")
            else:
                train_dataset_tensor = load_small_dataset_in_memory('test').share_memory_()
                print(f"test->{train_dataset_tensor.shape}")
                valid_dataset_tensor = None
        else:
            if args.mode=='fourcast':
                train_dataset_tensor = load_test_dataset_in_memory(years=[2018],
                                                                root="datasets/era5G32x64_new"
                                                                ).share_memory_()
                print(f"test->{train_dataset_tensor.shape}")
                valid_dataset_tensor = None
        print("=======done==========")

    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,train_dataset_tensor,valid_dataset_tensor))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, ngpus_per_node, args,train_dataset_tensor,valid_dataset_tensor)

if __name__ == '__main__':
    main()
