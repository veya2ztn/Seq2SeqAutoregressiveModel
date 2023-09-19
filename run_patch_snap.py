import torch
from cephdataset import *
from train.pretrain import *
from train.pretrain import get_args
from mltool.universal_model_util import get_model_para_detail
import numpy as np
import os
import sys  


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
    ltmv_pred_record = []
    target_record = []
    for i in range(model.history_length,len(batch)):# i now is the target index
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,batch[i],dataset,time_step_1_mode)
        if extra_loss !=0:
            iter_info_pool[f'{status}_extra_loss_gpu{gpu}_timestep{i}'] = extra_loss.item()
        for extra_info_from_model in extra_info_from_model_list:
            for name, value in extra_info_from_model.items():
                iter_info_pool[f'valid_on_{status}_{name}_timestep{i}'] = value
        
        ltmv_pred = dataset.do_normlize_data([ltmv_pred])[0]

        abs_loss = criterion(ltmv_pred,target)
        ltmv_pred_record.append(ltmv_pred)
        target_record.append(target)
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
    return loss, diff, iter_info_pool,torch.cat(ltmv_pred_record),torch.cat(target_record)


#ckpt_path = "checkpoints/WeathBench7066PatchDataset/PatchWrapper-AFNONet/time_step_2_pretrain-2D70N_every_1_step_random_dataset/11_11_05_03_43-seed_76545"
ckpt_path = sys.argv[1]
args=get_args(os.path.join(ckpt_path,"config.json"))

args.use_wandb=0
args.gpu = args.local_rank = gpu  = local_rank = 0
##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
args= parse_default_args(args)
#SAVE_PATH = get_ckpt_path(args)
#SAVE_PATH = "debug"
#args.SAVE_PATH = str(SAVE_PATH)
#args.Checkpoint.pretrain_weight = os.path.join(args.SAVE_PATH,'pretrain_latest.pt')
########## inital log ###################
logsys = create_logsys(args,False)


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
loss_scaler     = torch.cuda.amp.GradScaler(enabled=True)
logsys.info(f'use lr_scheduler:{lr_scheduler}')

pretrain_path = os.path.join(ckpt_path,"pretrain_latest.pt")
args.Checkpoint.pretrain_weight = pretrain_path

logsys.info(f"loading weight from {args.Checkpoint.pretrain_weight}")
start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, optimizer, lr_scheduler, loss_scaler, path=args.Checkpoint.pretrain_weight, 
                    only_model= (args.Train.mode=='fourcast') or (args.Train.mode=='finetune' and not args.Train.mode == 'continue_train') ,loc = 'cuda:{}'.format(args.gpu))
if args.more_epoch_train:
    assert args.Checkpoint.pretrain_weight
    print(f"detect more epoch training, we will do a copy processing for {args.Checkpoint.pretrain_weight}")
    os.system(f'cp {args.Checkpoint.pretrain_weight} {args.Checkpoint.pretrain_weight}-epoch{start_epoch}')
logsys.info("done!")

train_dataset, val_dataset, train_dataloader,val_dataloader = get_train_and_valid_dataset(args,
               train_dataset_tensor=None,train_record_load=None,
               valid_dataset_tensor=None,valid_record_load=None)
logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
logsys.info(f"Start training for {args.Train.epochs} epochs")
master_bar = logsys.create_master_bar(args.Train.epochs)
accu_list = ['valid_loss']
metric_dict = logsys.initial_metric_dict(accu_list)

train_dataset.cross_sample  =1 
train_dataloader = torch.utils.data.DataLoader(train_dataset,args.valid_batch_size)
val_dataset.cross_sample  =1 
val_dataloader = torch.utils.data.DataLoader(val_dataset,args.valid_batch_size)
for data_loader, datasetflag in zip([train_dataloader,val_dataloader],['train','valid']):
    epoch = 0
    start_step = 0
    status = 'valid'
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
    intervel = batches//100 + 1


    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = -1

    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")
    preds = []
    reals = []
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
        assert status != 'train'
        with torch.no_grad():
            #print([t.shape for t in batch])
            loss, abs_loss, iter_info_pool,pred,real =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
            if optimizer.grad_modifier is not None:
                Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
        iter_info_pool={}
        iter_info_pool[f'{status}_loss_gpu{gpu}']     =  loss.item()
        iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}'] = Nodeloss1
        iter_info_pool[f'{status}_Nodeloss12_gpu{gpu}'] = Nodeloss12
        iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}'] = Nodeloss2
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TimeLeng]:{time_step_now:} GPU:[{gpu}] abs_loss:{abs_loss.item():.2f} loss:{loss.item():.2f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
        #print(data_loader.dataset.record_load_tensor.mean().item())
        data_cost  = []
        train_cost = []
        rest_cost = []
        inter_b.lwrite(outstring, end="\r")
        preds.append(pred.detach().cpu())
        reals.append(real.detach().cpu())
        if step>10:break
    preds = torch.cat(preds)
    reals = torch.cat(reals)
    
    preds = preds.reshape(preds.shape[0],70,*preds.shape[-2:])
    reals = reals.reshape(reals.shape[0],70,*reals.shape[-2:])
    print(preds.shape)
    print(reals.shape)
    fig, axs = plt.subplots(14, 5,figsize=(60,30))
    for property_id in range(preds.shape[1]):
        pred = preds[:,property_id].flatten()
        real = reals[:,property_id].flatten()
        property_name = data_loader.dataset.vnames[property_id]
        ax = axs[property_id%14][property_id//14]
        order = torch.argsort(real)
        pred= pred[order]
        real= real[order]
        ax.plot(pred)
        ax.plot(real)
        ax.set_xticks([])
        ax.set_title(property_name)
        
    fig.savefig(os.path.join(logsys.ckpt_root,f"14x5_patch_error_image_{datasetflag}.png"))