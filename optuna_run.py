from inspect import ArgSpec
import optuna,os,time,copy
import os, sys,time
sys.path.append(os.getcwd())
from optuna.trial import TrialState
from train.pretrain import *
import torch
import numpy as np
import random
import argparse
batchsize_list      = [32,64,128]
lr_range            = [1e-3,1e-1]
patchsize_list      = [2,4,8]
grad_clip_list      = [1,1e2,1e4,None]
#input_noise_std_list= [0, 0.0001, 0.001, 0.01]
OPTUNALIM           = 10
error_time=0

def set_range_optuna_list(trials, args,range_string,flag):
    if range_string is None:return
    val_range = [float(t) for t in range_string.split(',')]
    if len(val_range)==1:
        val  = args.hparam_dict[flag] = val_range[0]
    else:
        assert len(val_range)==2
        val  = args.hparam_dict[flag] = trial.suggest_uniform(flag, *val_range)
    setattr(args,flag, val)

def set_select_optuna_list(trials, args,range_string,flag):
    if range_string is None:
        return
    val_range = [float(t) for t in range_string.split(',')]
    if len(val_range)==1:
        val  = args.hparam_dict[flag] = val_range[0]
    else:
        assert len(val_range)==2
        val  = args.hparam_dict[flag] = trial.suggest_categorical(flag, val_range)
    setattr(args,flag, val)

def optuna_high_level_main():
    conf_parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,add_help=False)
    conf_parser.add_argument("--batchsize_list",  default="4,16,32")
    conf_parser.add_argument("--lr_range",        default="0.0001,0.01")
    conf_parser.add_argument("--patchsize_list",  default="2")
    conf_parser.add_argument("--grad_clip_list",  default=None)
    conf_parser.add_argument("--batchsize_list",  default="4,16,32")
    conf_parser.add_argument("--optuna_trails",   default="3")
    optuna_args, remaining_argv = conf_parser.parse_known_args()
    gargs = get_args()
    #train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load = create_memory_templete(gargs)
    def objective(trial):
        args = copy.deepcopy(gargs)
        #args.distributed= False
        #args.rank=0
        #random_seed= args.seed
        args.seed  = random_seed= random.randint(1, 100000)

        args.hparam_dict = {}
        set_range_optuna_list(trials, args,optuna_args.lr_range,'lr')
        if gargs.batch_size==-1:
            set_select_optuna_list(trials, args,optuna_args.batch_size_list,'batch_size')
        if not gargs.clip_grad:
            set_select_optuna_list(trials, args,optuna_args.grad_clip_list,'grad_clip')
        if not gargs.patch_size:
            set_select_optuna_list(trials, args,optuna_args.patch_size_list,'patch_size')

        args.valid_batch_size = args.batch_size
        
        # if not gargs.input_noise_std:
        #     args.input_noise_std = args.hparam_dict['input_noise_std'] = trial.suggest_categorical("input_noise_std", input_noise_std_list)
        # #trial.set_user_attr('trial_name', TRIAL_NOW)

        #################################################################################
        #result=main(args)
        
        # args = distributed_initial(args)
        
        # result_tensor = torch.zeros(1).share_memory_()
        # if args.multiprocessing_distributed:
        #     print("======== entering  multiprocessing train ==========")
        #     args.world_size = args.ngpus_per_node * args.world_size
        #     torch.multiprocessing.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args,result_tensor,
        #                                 train_dataset_tensor,train_record_load,
        #                                 valid_dataset_tensor,valid_record_load))
        # else:
        #     print("======== entering  single gpu train ==========")
        #     main_worker(0, args.ngpus_per_node, args,result_tensor,
        #     train_dataset_tensor,train_record_load,valid_dataset_tensor,valid_record_load)
        
        # result = {'valid_loss':result_tensor.mean().item()}
        # torch.cuda.empty_cache()

        print(args)
        result = np.random.randn()
        #################################################################################
        timenow     = time.asctime( time.localtime(time.time()))
        result_string = " ".join([f"{key}:{np.mean(val)}" for key,val in result.items()])
        info_string = f"{timenow} {result_string}\n"
        with open("exp_hub",'a') as f:f.write(info_string)

        the_key = [k for k in result.keys() if 'valid' in k][0]
        return result[the_key]
    ########## optuna high level script  ###########
    model_name, datasetname, project_name = get_projectname(gargs)
    DB_NAME   = f"{datasetname}-{model_name}-{project_name}"

    TASK_NAME = 'task1'
    study = optuna.create_study(study_name=TASK_NAME, storage=f'sqlite:///optuna_database/{DB_NAME}.db',
                                    load_if_exists=True,
                                    sampler=optuna.samplers.CmaEsSampler(),
                                    pruner =optuna.pruners.MedianPruner(n_warmup_steps=28)
                                )
    # optuna_limit_trials = OPTUNALIM
    # if len([t.state for t in study.trials if t.state== TrialState.COMPLETE])>optuna_limit_trials:
    #     return 'up tp optuna setted limit'
    #study.optimize(objective, n_trials=50, timeout=600,pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=optuna_args.optuna_trails,  gc_after_trial=True)

if __name__ == '__main__':
    optuna_high_level_main()
