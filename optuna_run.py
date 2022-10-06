from inspect import ArgSpec
import optuna,os,time,copy
import os, sys,time
sys.path.append(os.getcwd())
from optuna.trial import TrialState
from train.pretrain import *
import torch
import numpy as np
import random

batchsize_list      = [16,32,64]
lr_range            = [1e-4,1e-3]
patchsize_list      = [2,4,8]
input_noise_std_list= [0, 0.0001, 0.001, 0.01]
OPTUNALIM           = 10
error_time=0

def optuna_high_level_main():
    gargs = get_args()
    def objective(trial):

        args = copy.deepcopy(gargs)
        args.distributed= False
        args.rank=0
        #random_seed= args.seed
        args.seed  = random_seed= random.randint(1, 100000)
        args.hparam_dict = {}
        args.lr             = args.hparam_dict['lr']         = trial.suggest_uniform(f"lr", *lr_range)
        if not gargs.batch_size:
            args.batch_size     = args.hparam_dict['batch_size'] = trial.suggest_categorical("batch_size", batchsize_list)
        if not gargs.patch_size:
            args.patch_size     = args.hparam_dict['patch_size'] = trial.suggest_categorical("patch_size", patchsize_list)
        # if not gargs.input_noise_std:
        #     args.input_noise_std = args.hparam_dict['input_noise_std'] = trial.suggest_categorical("input_noise_std", input_noise_std_list)
        # #trial.set_user_attr('trial_name', TRIAL_NOW)

        #################################################################################
        train_dataset_tensor=valid_dataset_tensor=train_record_load=valid_record_load=None
        if args.use_inmemory_dataset:
            assert args.dataset_type
            print("======== loading data as shared memory==========")
            if not args.mode=='fourcast':
                print(f"create training dataset template, .....")
                train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='train',root=args.data_root)
                train_dataset_tensor = train_dataset_tensor.share_memory_()
                train_record_load  = train_record_load.share_memory_()
                print(f"done! -> train template shape={train_dataset_tensor.shape}")
                
                print(f"create validing dataset template, .....")
                valid_dataset_tensor, valid_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='valid',root=args.data_root)
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

        result=main_worker(0, 1, args,train_dataset_tensor,valid_dataset_tensor)
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
    study.optimize(objective, n_trials=5,gc_after_trial=True)
if __name__ == '__main__':
    optuna_high_level_main()
