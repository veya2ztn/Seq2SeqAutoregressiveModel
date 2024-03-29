
from train.epoch_step import run_one_epoch
from train.nodal_snap_step import run_nodalosssnap
from evaluator.evaluate import run_fourcast, run_fourcast_during_training
import numpy as np
from configs.arguments import get_args
from utils.tools import save_state, load_model, get_local_rank, dprint
import torch.distributed as dist
import os
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs
from configs.utils import get_ckpt_path
from utils.loggingsystem import create_logsys
from model.get_resource import build_training_resource
from dataset.get_resource import get_test_dataset, get_train_and_valid_dataset
from train.sequence2sequence_manager import FieldsSequence
#########################################
############# main script ###############
#########################################



def fast_set_model_epoch(model, criterion, dataloader, optimizer, loss_scaler, epoch, args, **kargs):
    unwrapper_model = model
    while hasattr(unwrapper_model, 'module'):
        unwrapper_model = unwrapper_model.module
    model = unwrapper_model
    if hasattr(model, 'set_epoch'):model.set_epoch(**kargs)

    if args.Pengine.engine.name == 'naive_distributed' and args.Pengine.engine.distributed and not args.Train.train_not_shuffle:
        dataloader.sampler.set_epoch(epoch)

def step_the_scheduler(lr_scheduler, now_lr, epoch, args):
    freeze_learning_rate = (args.Scheduler.scheduler_min_lr and now_lr < args.Scheduler.scheduler_min_lr) and (
        args.Scheduler.scheduler_inital_epochs and epoch > args.Scheduler.scheduler_inital_epochs)
    if args.Train.mode not in ['more_epoch_train'] and (lr_scheduler is not None) and not freeze_learning_rate:lr_scheduler.step(epoch)

def update_and_save_training_status(args, epoch, loss_information, training_state):
    if get_local_rank(args):return 
    ### loss_information:
    ## {'train_loss': {'now':0, 'best':0, 'save_path':.....},
    ##  'valid_loss': {'now':0, 'best':0, 'save_path':.....},
    ##  'test_loss' : {'now':0, 'best':0, 'save_path':.....}}
    
    logsys = args.logsys
    logsys.metric_dict.update({'valid_loss':loss_information['valid_loss']['now']},epoch)
    logsys.banner_show(epoch,args.SAVE_PATH,train_losses=[loss_information['train_loss']['now']])

    
    for loss_type in ['valid_loss', 'test_loss']:
        now_loss = loss_information[loss_type]['now']
        if now_loss is None:continue
        best_loss = loss_information[loss_type]['best']
        save_path = loss_information[loss_type]['save_path']
        ####### save best valid model #########
        if now_loss < best_loss or best_loss is None:
            loss_information[loss_type]['best'] = best_loss = now_loss
            if epoch > args.Train.epochs//10:
                logsys.info(f"saving best model for {loss_type}....",show=False)
                performance = dict([(name, val['best']) for name, val in loss_information])
                save_state(epoch=epoch, path=save_path, only_model=True, performance=performance, **training_state)
                logsys.info(f"done;",show=False)
            logsys.info(f"The best {loss_type} is {best_loss}", show=False)
            logsys.record(f'best_{loss_type}', best_loss, epoch, epoch_flag='epoch')
        

    ###### save runtime checkpoints #########
    #update_experiment_info(experiment_hub_path,epoch,args)
    loss_type = 'train_loss'
    now_loss = loss_information[loss_type]['now']
    best_loss = loss_information[loss_type]['best']
    save_path = loss_information[loss_type]['save_path']
    loss_information[loss_type]['best'] = loss_information[loss_type]['now']
    save_path = loss_information[loss_type]['save_path']
    if ((epoch>=args.Checkpoint.save_warm_up) and (epoch%args.Checkpoint.save_every_epoch==0)) or (epoch==args.Train.epochs-1) or (epoch in args.Checkpoint.epoch_save_list):
        logsys.info(f"saving latest model ....", show=False)
        save_state(epoch=epoch, step=0, path=save_path, only_model=False, performance=performance, **training_state)
        logsys.info(f"done ....",show=False)
        if epoch in args.Checkpoint.epoch_save_list:
            save_state(epoch=epoch, path=f'{save_path}-epoch{epoch}', only_model=True, performance=performance, **training_state)
    return loss_information
    

def distributed_runtime(config, local_rank):
    if config.name != 'naive_distributed':return 
    config.local_rank = local_rank
    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * config.ngpus_per_node + config.local_rank
        print(f"start init_process_group,backend={config.dist_backend}, init_method={config.dist_url},world_size={config.world_size}, rank={config.rank}")
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,world_size=config.world_size, rank=config.rank)



def build_accelerator(args):
    accelerator = None
    if args.Pengine.engine.name == 'accelerate':
        project_config = ProjectConfiguration(
            project_dir=str(args.SAVE_PATH),
            automatic_checkpoint_naming=True,
            total_limit=args.Pengine.engine.num_max_checkpoints,
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.Pengine.engine.find_unused_parameters)
        accelerator = Accelerator(dispatch_batches=not args.Pengine.engine.data_parallel_dispatch,
                                  project_config=project_config,
                                  log_with=None, kwargs_handlers=[ddp_kwargs]
                                )
    return accelerator

from tqdm.auto import tqdm
import torch
def main_worker(local_rank, ngpus_per_node, args,
                result_tensor=None,
                train_dataset_tensor=None,train_record_load=None,
                valid_dataset_tensor=None,valid_record_load=None):
    """
    xxxx_dataset_tensor used for shared-in-memory dataset among DDP
    """
    distributed_runtime(args.Pengine.engine, local_rank)
    args.gpu = get_local_rank(args)
    ##### locate the checkpoint dir ###########
    #--- parse args: dataset_kargs / model_kargs / train_kargs  #---
    #args      = parse_default_args(args)
    args.SAVE_PATH = get_ckpt_path(args)
    ########## inital log ###################
    logsys = create_logsys(args)
    args.logsys = logsys
    #########################################
    args.accelerator = build_accelerator(args)

    ####### Initialize Training Resource #########
    model, optimizer, lr_scheduler, criterion, loss_scaler = build_training_resource(args)

    logsys.info(f"entering {args.Train.mode} training in {next(model.parameters()).device}")
    if args.Train.mode=='fourcast':
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=train_dataset_tensor,test_record_load=train_record_load)
        run_fourcast(args, model,logsys,test_dataloader)
        return logsys.close()
    elif args.Train.mode=='fourcast_for_snap_nodal_loss':
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=train_dataset_tensor,test_record_load=train_record_load)
        run_nodalosssnap(args, model,logsys,test_dataloader)
        return logsys.close()
    
    ####### Initialize Dataset #########
    
    train_dataset, valid_dataset, train_dataloader,valid_dataloader = get_train_and_valid_dataset(args,
                    train_dataset_tensor=train_dataset_tensor,train_record_load=train_record_load,
                    valid_dataset_tensor=valid_dataset_tensor,valid_record_load=valid_record_load)
    test_dataloader = None 

    ####### Initialize Sequence Controller #########

    sequence_manager      = FieldsSequence(args)

    start_step  = args.start_step
    start_epoch = args.start_epoch

    logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
    logsys.info(f"Start training for {args.Train.epochs} epochs")
    
    master_bar = logsys.create_master_bar(list(range(-1, args.Train.epochs)))
    accu_list = ['valid_loss']
    metric_dict = logsys.initial_metric_dict(accu_list)
    banner = logsys.banner_initial(args.Train.epochs, args.SAVE_PATH)
    logsys.banner_show(0, args.SAVE_PATH)
    loss_information = {'train_loss': {'now': np.inf, 'best': np.inf, 'save_path':os.path.join(args.SAVE_PATH,'pretrain_latest.pt')},
                        'test_loss' : {'now': np.inf, 'best': np.inf, 'save_path': os.path.join(args.SAVE_PATH, 'fourcast.best.pt')},
                        'valid_loss': {'now': np.inf, 'best': args.min_loss, 'save_path': os.path.join(args.SAVE_PATH, 'backbone.best.pt')},
                        }
    
    train_loss = np.inf
    for epoch in master_bar:
        if epoch < start_epoch:continue

        ## skip fisrt training, and the epoch should be -1
        if epoch >=0: ###### main training loop #########
            fast_set_model_epoch(model, criterion, train_dataloader, optimizer, loss_scaler,epoch, args, epoch_total=args.Train.epochs, eval_mode=False)
            
            training_system = {'model':model, 'criterion':criterion, 'optimizer':optimizer, 'loss_scaler':loss_scaler,
                               'use_amp': (args.Pengine.engine.name == 'naive_distributed' and args.Pengine.engine.use_amp), 
                               'accumulation_steps': args.Train.accumulation_steps}
            train_loss = run_one_epoch('train', epoch, start_step,  train_dataloader,  training_system, sequence_manager, args.logsys, args.accelerator, plugins=[])
            logsys.record('train', train_loss, epoch, epoch_flag='epoch')
            loss_information['train_loss']['now'] = train_loss
            learning_rate = optimizer.param_groups[0]['lr']
            logsys.record('learning rate',learning_rate,epoch, epoch_flag='epoch')
            step_the_scheduler(lr_scheduler, optimizer.param_groups[0]['lr'], epoch, args)  # make sure the scheduler is load from state

        
        if (args.Forecast.forecast_every_epoch and 
            ((epoch % args.Forecast.forecast_every_epoch == 0) or 
            (epoch == -1 and args.force_do_first_fourcast) or 
            (epoch == args.Train.epoch - 1))):
            
            fast_set_model_epoch(model,criterion, valid_dataloader, optimizer, loss_scaler,epoch, args, epoch_total=args.Train.epochs,eval_mode=True)
            test_loss, test_dataloader = run_fourcast_during_training(args, epoch, logsys, model, test_dataloader)  # will
            loss_information['test_loss']['now'] = test_loss
            logsys.record('test', test_loss, epoch, epoch_flag='epoch') 
            
        if (args.Valid.valid_every_epoch and 
            ((epoch % args.Valid.valid_every_epoch == 0) or 
            (epoch == -1 and not args.Valid.skip_first_valid) or 
            (epoch == args.Train.epoch - 1))):
            
            fast_set_model_epoch(model,criterion, valid_dataloader, optimizer, loss_scaler,epoch, args, epoch_total=args.Train.epochs,eval_mode=True)
            validation_system = {'model':model, 'criterion':criterion}
            val_loss   = run_one_epoch('valid', epoch, None,  valid_dataloader,  validation_system, sequence_manager, args.logsys, args.accelerator, plugins=[])
            loss_information['valid_loss']['now'] = val_loss
            logsys.record('valid', val_loss, epoch, epoch_flag='epoch')
        
        update_and_save_training_status(args, epoch, loss_information, {'model':model, 'criterion':criterion, 
                                                                        'optimizer':optimizer,'loss_scaler':loss_scaler})
        logsys.info(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}", show=False)
    
        
    best_valid_ckpt_path = loss_information['valid_loss']['save_path']
    if os.path.exists(best_valid_ckpt_path) and args.do_final_fourcast:
        logsys.info(f"we finish training, then start test on the best checkpoint {best_valid_ckpt_path}")
        args.Train.mode = 'fourcast'
        args.Dataset.time_step = 22
        start_epoch, start_step, min_loss = load_model(model.module if args.multiprocess_distributed else model, path=best_valid_ckpt_path, only_model=True, loc='cuda:{}'.format(args.gpu))
        run_fourcast(args, model,logsys)
        
    if result_tensor is not None and local_rank==0:result_tensor[local_rank] = min_loss
    return logsys.close()


if __name__ == '__main__':
    args = get_args()
    main_worker(0,0, args)
