from inspect import ArgSpec
import optuna,os,time,copy
import os, sys,time
sys.path.append(os.getcwd())
from optuna.trial import TrialState
from train.pretrain import *
import torch
import numpy as np
import random

batchsize_list      = [32,64,128]
lr_range            = [1e-3,1e-1]
patchsize_list      = [2,4,8]
grad_clip_list      = [1,1e2,1e4,None]
input_noise_std_list= [0, 0.0001, 0.001, 0.01]
OPTUNALIM           = 10
error_time=0

def optuna_high_level_main():
    gargs = get_args()
    train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load = create_memory_templete(gargs)
    def objective(trial):
        args = copy.deepcopy(gargs)
        #args.distributed= False
        #args.rank=0
        #random_seed= args.seed
        args.seed  = random_seed= random.randint(1, 100000)
        args.hparam_dict = {}
        args.lr        = args.hparam_dict['lr']         = trial.suggest_uniform(f"lr", *lr_range)
        if gargs.batch_size==-1:
            args.batch_size = args.hparam_dict['batch_size'] = trial.suggest_categorical("batch_size", batchsize_list)
        # if not gargs.clip_grad:
        #     args.clip_grad = args.hparam_dict['clip_grad'] = trial.suggest_categorical("clip_grad", grad_clip_list)
        # if not gargs.patch_size:
        #     args.patch_size     = args.hparam_dict['patch_size'] = trial.suggest_categorical("patch_size", patchsize_list)
        args.valid_batch_size = args.batch_size
        args.patch_size  = 2 
        print("notice we will fix patch size as 2")
        # if not gargs.input_noise_std:
        #     args.input_noise_std = args.hparam_dict['input_noise_std'] = trial.suggest_categorical("input_noise_std", input_noise_std_list)
        # #trial.set_user_attr('trial_name', TRIAL_NOW)

        #################################################################################
        #result=main(args)
        
        args = distributed_initial(args)
        
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
        
        result = {'valid_loss':result_tensor.mean().item()}
        torch.cuda.empty_cache()
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
    study.optimize(objective, n_trials=20,  gc_after_trial=True)
if __name__ == '__main__':
    optuna_high_level_main()
