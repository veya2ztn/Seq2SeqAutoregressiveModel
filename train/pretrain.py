
import os, sys,time
sys.path.append(os.getcwd())
idx=0
sys.path = [p for p in sys.path if 'lustre' not in p]

force_big  = True
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
from model.physics_model import EulerEquationModel
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import single_step_evaluate


from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.loggingsystem import LoggingSystem


save_intervel=100
from cephdataset import ERA5CephDataset,ERA5CephSmallDataset,SpeedTestDataset,load_test_dataset_in_memory,load_small_dataset_in_memory,ERA5CephInMemoryDataset
#dataset_type = ERA5CephDataset
# dataset_type  = SpeedTestDataset

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def train_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss,logsys):
    model.train()
    logsys.train()

    accumulation_steps = 1 #8 # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16
    now = time.time()
    data_cost = train_cost = rest_cost = 0

    Fethcher   = DataSimfetcher
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    batches    = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0
    inter_b.lwrite("load everything, start_training......", end="\r")
    if start_step == 0:optimizer.zero_grad()
    while inter_b.update_step():
        step = inter_b.now
        batch = prefetcher.next()
        if step < start_step:continue

        data_cost += time.time() - now;now = time.time()

        batch = [x.half() if half_model else x.float() for x in batch] 
        batch = [x.to(device).transpose(3, 2).transpose(2, 1) for x in batch]
        loss = 0
        
        with torch.cuda.amp.autocast():
            out = batch[0]
            for i in range(1,len(batch)):
                out   = model(out)
                extra_loss = 0
                if isinstance(out,(list,tuple)):
                    extra_loss=out[1]
                    logsys.record(f'train_extra_loss_gpu{gpu}_timestep{i}', extra_loss.item(), epoch*batches + step)
                    for extra_info_from_model in out[2:]:
                        for name, value in extra_info_from_model.items():
                            logsys.record(f'train_{name}_timestep{i}', value, epoch*batches + step)
                    out = out[0]
                loss += criterion(out, batch[i]) + extra_loss
            loss /= accumulation_steps
        loss_scaler.scale(loss).backward()
        
        train_cost += time.time() - now;now = time.time()
        logsys.record(f'train_training_loss_gpu{gpu}', loss.item(), epoch*batches + step)
        # 梯度累积
        if (step+1) % accumulation_steps == 0:
           #if half_model:
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()
            # else:
            #     optimizer.step()
            #     optimizer.zero_grad()

        # if (step+1) % save_intervel == 0:
        #     #if dist.get_rank() == 0 and hfai.receive_suspend_command():
        #     if hasattr(model,'module'):
        #         if gpu==0:
        #             save_model(model, epoch, step+1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH/'pretrain_latest.pt')
        #         #time.sleep(5)
        #     #hfai.go_suspend()
        #     else:
        #         save_model(model, epoch, step+1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH/'pretrain_latest.pt')

        rest_cost += time.time() - now;now = time.time()
        intervel=10
        if (step+1) % intervel==0 or step==0:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}] loss:{loss.item():.2f} cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} [Rest]:{rest_cost/intervel:.1e}")
            data_cost = train_cost = rest_cost = 0
            inter_b.lwrite(outstring, end="\r")
        
def single_step_evaluate(data_loader, model, criterion,epoch,logsys):
    loss, total = torch.zeros(2).cuda()
    gpu     = dist.get_rank() if hasattr(model,'module') else 0
    # switch to evaluation mode
    model.eval()
    logsys.eval()
    Fethcher   = DataSimfetcher
    device = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,next(model.parameters()).device)
    batches = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost = train_cost = rest_cost = 0
    now = time.time()

    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            data_cost += time.time() - now;now = time.time()
            step = inter_b.now
            batch = prefetcher.next()
            batch = [x.half() if half_model else x.float() for x in batch] 
            batch = [x.to(device).transpose(3, 2).transpose(2, 1) for x in batch]
            with torch.cuda.amp.autocast():
                out = batch[0]
                for i in range(1,len(batch)):
                    out   = model(out)
                    extra_loss = 0
                    if isinstance(out,(list,tuple)):
                        extra_loss=out[1]
                        logsys.record(f'valid_extra_loss_gpu{gpu}_timestep{i}', extra_loss.item(), epoch*batches + step)
                        for extra_info_from_model in out[2:]:
                            for name, value in extra_info_from_model.items():
                                logsys.record(f'valid_{name}_timestep{i}', value, epoch*batches + step)
                        out = out[0]
                    loss += criterion(out, batch[i]) + extra_loss
            train_cost += time.time() - now;now = time.time()
            #total += 1
            total += len(batch) - 1
            rest_cost += time.time() - now;now = time.time()
            intervel=10
            if (step+1) % intervel==0 or step==0:
                outstring=(f"epoch:valid iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}] loss:{loss.item():.2f} cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} [Rest]:{rest_cost/intervel:.1e}")
                inter_b.lwrite(outstring, end="\r")


        if hasattr(model,'module'):
            for x in [loss, total]:
                dist.barrier()
                dist.reduce(x, 0)

        if hasattr(model,'module'):
            loss_val = 0
            if dist.get_rank() == 0:
                loss_val = loss.item() / total.item()
            return loss_val
        else:
            return loss.item() / total.item()
        

def compute_accu(ltmsv_pred, ltmsv_true):
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
    return fenzi/(fenmu+1e-10)

def compute_rmse(pred, true):
    # pred <-- (B,w,h,p)
    # true <-- (B,w,h,p)
    w         = pred.shape[1]
    latitude  = torch.linspace(-np.pi/2, np.pi/2, w).to(pred.device)
    cos_lat   = torch.cos(latitude)
    latweight = cos_lat/cos_lat.mean()
    latweight = latweight.reshape(1, w, 1,1)
    return  torch.sqrt((latweight*(pred - true)**2).mean(dim=(1,2) ))

def fourcast_step(data_loader, model,logsys,random_repeat = 0):
    model.eval()
    logsys.eval()
    gpu     = dist.get_rank() if hasattr(model,'module') else 0
    Fethcher   = DataSimfetcher
    prefetcher = Fethcher(data_loader,next(model.parameters()).device)
    batches = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    device = next(model.parameters()).device
    data_cost = train_cost = rest_cost = 0
    now = time.time()

    fourcastresult={}
    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            data_cost += time.time() - now;now = time.time()
            step = inter_b.now
            idxes,batch = prefetcher.next()
            batch = [x.half() if half_model else x.float() for x in batch] 
            batch = [x.to(device) for x in batch] 
            if batch[0].shape[1]!=20:batch = [x.permute(0,3,1,2) for x in batch]
        #for idxes,batch in tqdm(data_loader):
            
            history_sum_true = history_sum_pred = batch[0].permute(0,2,3,1) if batch[0].shape[1]==20 else batch[0]

            accu_series=[]
            rmse_series=[]
            
            out = batch[0]
            for i in range(1,len(batch)):
                out   = model(out)
                extra_loss = 0
                if isinstance(out,(list,tuple)):
                    extra_loss=out[1]
                    out = out[0]
                ltmv_pred = out.permute(0,2,3,1)
                ltmv_true = batch[i].permute(0,2,3,1)
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
                #if idx in fourcastresult:print(f"repeat at idx={idx}")
                fourcastresult[idx.item()] = {'accu':accu,"rmse":rmse}
            
             
            for _ in range(random_repeat):
                out = batch[0]*(1 + torch.randn_like(batch[0])*0.05)
                for i in range(1,len(batch)):
                    out   = model(out)
                    ltmv_pred = out.permute(0,2,3,1)
                    ltmv_true = batch[i].permute(0,2,3,1)
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
                    if 'random' in fourcastresult[idx.item()]:fourcastresult[idx.item()]['random']={'accu':[],"rmse":[]}
                    fourcastresult[idx.item()]['random']['accu'].append(accu)
                    fourcastresult[idx.item()]['random']['rmse'].append(rmse)


    save_path = os.path.join(logsys.ckpt_root,f"fourcastresult.gpu_{gpu}")
    torch.save(fourcastresult,save_path)
    print(f"save fourcastresult at {save_path}")
lr_for_mode={
    'pretrain':5e-4,
    'finetune':1e-4
}
ep_for_mode={
    'pretrain':80,
    'finetune':50
}
bs_for_mode={
    'pretrain':4,
    'finetune':3
}
as_for_mode={
    'pretrain':8,
    'finetune':16
}
train_set={
    'large': (720, 1440, 8, ERA5CephDataset),
    'small': ( 32,   64, 2, ERA5CephSmallDataset),
    'test_large': (720, 1440, 8, lambda **kargs:SpeedTestDataset(720,1440,**kargs)),
    'test_small': ( 32,   64, 8, lambda **kargs:SpeedTestDataset( 32,  64,**kargs))
}

half_model = False

last_best_path=None
def main_worker(local_rank, ngpus_per_node, args, train_dataset_tensor=None,valid_dataset_tensor=None):
    args.activate_physics_dataset = True
    args.activate_physics_model = True
    TIME_NOW  = time.strftime("%m_%d_%H_%M")
    if not hasattr(args,'train_set'):args.train_set='large'
    #TIME_NOW = "07_23_19_35_04"
    name = args.mode if not args.fourcast else 'fourcast'
    SAVE_PATH = Path(f'./checkpoints/fourcastnet/{name}-{args.train_set}/{TIME_NOW}')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    args.gpu = gpu = local_rank
    args.half_model = half_model
    args.batch_size = bs_for_mode[args.mode] if args.batch_size == -1 else args.batch_size
    #args.batch_size = 1 if force_big else 1
    ngpus = torch.cuda.device_count()
    args.epochs = ep_for_mode[args.mode] if args.epochs == -1 else args.epochs
    args.lr     = lr_for_mode[args.mode] if args.lr == -1 else args.lr
    # input size
    h, w, patch_size,dataset_type = train_set[args.train_set] 
    x_c, y_c = 20, 20
    if args.activate_physics_dataset:
        x_c, y_c = 12, 12
    if args.activate_physics_model:
        y_c  = 15
    logsys   = LoggingSystem(local_rank==0,SAVE_PATH,seed=args.seed)
    _        = logsys.create_recorder(hparam_dict={},metric_dict={})
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #cudnn.benchmark = True
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        print(f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

    if args.fourcast:
        if 'small' in args.train_set:
            dataset_type  = ERA5CephInMemoryDataset
            test_dataset  = dataset_type(split="test" , mode=args.mode,years=[2018],root="/nvme/zhangtianning/datasets/ERA5", 
                                            check_data=True,
                                            dataset_tensor=train_dataset_tensor, 
                                            with_idx=True,
                                            time_step=9*24//6,enable_physics_dataset=args.activate_physics_dataset)
        else:
            test_dataset = dataset_type(split="test", mode=args.mode, check_data=True,dataset_tensor=train_dataset_tensor,
            time_step=9*24//6,enable_physics_dataset=args.activate_physics_dataset)
    else:
        train_dataset = dataset_type(split="train", mode=args.mode, check_data=True,dataset_tensor=train_dataset_tensor,
                                    enable_physics_dataset=args.activate_physics_dataset)
        val_dataset   = dataset_type(split="valid", mode=args.mode, check_data=True,dataset_tensor=valid_dataset_tensor,
                                    enable_physics_dataset=args.activate_physics_dataset)
    print(f"use dataset ==> {dataset_type.__name__}")

    #train_dataset.file_list=train_dataset.file_list[:20]
    #val_dataset.file_list=val_dataset.file_list[:20]

    if args.fourcast:
        test_datasampler  = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
    else:
        train_datasampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
        val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None

    if args.fourcast:
        test_dataloader    = DataLoader(test_dataset, 1, sampler=test_datasampler, num_workers=8, pin_memory=False,drop_last=False)
    else:
        train_dataloader  = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)
        val_dataloader    = DataLoader(val_dataset, args.batch_size*8, sampler=val_datasampler, num_workers=8, pin_memory=True, drop_last=False)
    if args.distributed or force_big:
        model = AFNONet(img_size=[h, w], patch_size=patch_size, in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        fno_blocks=args.fno_blocks,
                        double_skip=args.double_skip, fno_bias=args.fno_bias, fno_softshrink=args.fno_softshrink,
                        )
    else:
        model = AFNONet(img_size=[h, w], patch_size=patch_size, in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        fno_blocks=args.fno_blocks,
                        double_skip=args.double_skip, fno_bias=args.fno_bias, fno_softshrink=args.fno_softshrink,
                        embed_dim=16, depth=1,debug_mode=1
                        )
    if args.activate_physics_model:
        model = EulerEquationModel(args,model)
    #model = hfnn.to_hfai(model)

    rank = args.rank
    if local_rank == 0:
        param_sum, buffer_sum, all_size = getModelSize(model)
        print(f"Rank: {args.rank}, Local_rank: {local_rank} | Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")
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
            assert args.pretrain_weight != ""
            assert os.path.exists(args.pretrain_weight)
            logsys.info(f"loading weight from {args.pretrain_weight}")
            start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, path=args.pretrain_weight, only_model=True)
            logsys.info("done!")
    #start_step = 0
    if args.fourcast:
        print("starting fourcast~!")
        
        with open(os.path.join(logsys.ckpt_root,'weight_path'),'w') as f:
            f.write(args.pretrain_weight)
        fourcast_step(test_dataloader, model,logsys,random_repeat = args.fourcast_randn_initial)
        return 1
    else:
        if local_rank == 0:
            print(f"Start training for {args.epochs} epochs")

        master_bar        = logsys.create_master_bar(args.epochs)
        last_best_path = None
        for epoch in master_bar:
            if epoch < start_epoch:continue
            train_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler,lr_scheduler, min_loss,logsys)
            lr_scheduler.step(epoch)
            #torch.cuda.empty_cache()
            #train_loss = single_step_evaluate(train_dataloader, model, criterion,logsys)
            train_loss = -1
            val_loss   = single_step_evaluate(val_dataloader, model, criterion,epoch,logsys)

            if rank == 0 and local_rank == 0:
                print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                logsys.record('train', train_loss, epoch)
                logsys.record('valid', val_loss, epoch)
                if val_loss < min_loss:
                    min_loss = val_loss
                    print(f"saving best model ....")
                    now_best_path = SAVE_PATH / f'backbone.best.pt'
                    save_model(model, path=now_best_path, only_model=True)
                    #if last_best_path is not None:os.system(f"rm {last_best_path}")
                    #last_best_path= now_best_path
                    print(f"done; the best accu is {val_loss}")
                print(f"saving latest model ....")
                start_step=0
                save_model(model, epoch+1, start_step, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')
                print(f"done ....")
        return 1

if __name__ == '__main__':
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
    
    print("======== loading data ==========")
    if 'small' in args.train_set:
        if not args.fourcast:
            train_dataset_tensor = load_small_dataset_in_memory('train').share_memory_()
            valid_dataset_tensor = load_small_dataset_in_memory('valid').share_memory_()
        else:
            train_dataset_tensor = load_small_dataset_in_memory('test').share_memory_()
            valid_dataset_tensor = None
    else:
        if args.fourcast:
            train_dataset_tensor = load_test_dataset_in_memory(years=[2018],root="/nvme/zhangtianning/datasets/ERA5").share_memory_()
            valid_dataset_tensor = None
    print("=======done==========")
    print(train_dataset_tensor.shape)
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,train_dataset_tensor,valid_dataset_tensor))
    else:
        main_worker(0, ngpus_per_node, args,train_dataset_tensor,valid_dataset_tensor)
