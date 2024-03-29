from ast import Not
from distutils.log import info
import os

import numpy as np
import argparse,tqdm,json,re
import sys
import pandas as pd
#wandb.init(settings=wandb.Settings(start_method="fork"))
datadir = "checkpoints/ERA5_20-12"

def remove_weight(trial_path):
    for file_name in ['pretrain_latest.pt','backbone.best.pt']:
        if  file_name in os.listdir(trial_path):

            weight_path = os.path.join(trial_path,file_name)
            if os.path.islink(weight_path):continue
            #if not os.path.islink(weight_path):
            ceph_path   = os.path.join("~/cephdrive/FourCastNet",weight_path)
            os.system(f"rm {weight_path}")
            #time.sleep(0.4)
            os.system(f"ln -s {ceph_path} {weight_path}")

def assign_trail_job(trial_path,wandb_id=None, gpu=0):
    from tbparse import SummaryReader
    import wandb
    from train.pretrain import create_logsys
    args              = get_the_args(trial_path)
    if args is None:return
    args.use_wandb    = 'wandb_runtime'
    args.wandb_id     = wandb_id
    #args.wandb_resume = 'must'
    args.gpu          = gpu
    args.recorder_list = []
    logsys = create_logsys(args, save_config=False)
    epoch_pool  = {}
    test_pool   = {}
    iter_metric = []
    hparams1    = hparams2=None
    mode        =  'pretrain' if 'pretrain' in trial_path else 'finetune'
    pattern     = re.compile(f'time_step_(.*)_{mode}')   # 查找数字
    time_step   = pattern.findall(trial_path)
    if len(time_step)==0:
        time_step=2
    else:
        time_step=int(time_step[0])

    summary_dir = trial_path
    for filename in os.listdir(summary_dir):
        if 'event' not in filename:continue
        log_dir = os.path.join(summary_dir,filename)
        reader = SummaryReader(log_dir)
        df = reader.scalars
        if 'tag' not in reader.hparams:
            print(f"no hparams at {log_dir}")
            #print(reader.hparams)
          
        if len(df) < 1: 
            print(f"no scalars at {log_dir},pass")
            continue
        print("start parsing tensorboard..............")
        for key in set(df['tag'].values):
            all_pool = test_pool if 'test' in key else epoch_pool
            if len(df[df['tag'] == key] )>1e3:
                #print(f"key={key} has too many summary, skip")
                #continue
                now = df[df['tag'] == key]

                steps = now['step'].values
                values= now['value'].values
                unipool={}
                for step, val in zip(steps,values):
                    if step not in unipool:unipool[step]=[]
                    unipool[step].append(val)
                steps=[]
                values=[]
                for step,val_list in unipool.items():
                    steps.append(step)
                    values.append(min(val_list))
                steps = np.array(steps)
                values= np.array(values)
                sortorder = np.argsort(steps)
                steps = steps[sortorder]
                values= values[sortorder]
                iter_metric.append([key, steps,values])
                
                #all_pool = iter_pool
                if 'tag' not in reader.hparams:
                    hparams1={}
                else:
                    hparams1  = dict([(name,v) for name,v in zip(reader.hparams['tag'].values,reader.hparams['value'].values)])
                continue
            if 'tag' not in reader.hparams:
                hparams2={}
            else:
                hparams2 = dict([(name,v) for name,v in zip(reader.hparams['tag'].values,reader.hparams['value'].values)])
            now = df[df['tag'] == key]
            steps = now['step'].values
            values= now['value'].values
            for step, val in zip(steps,values):
                if step not in all_pool:all_pool[step]={}
                all_pool[step][key]=val
    print("tensorboard parse done, start wandb..............")
    hparams = hparams2 if hparams1 is None else hparams1
    if hparams == None:return
    hparams['mode']      = mode
    hparams['time_step'] = time_step
    with open(os.path.join(trial_path,'config.json'), 'r') as f:
        hparams = json.load(f)

    
    for step, record in epoch_pool.items():
        record['epoch'] = int(step)
        logsys.wandblog(record)
    for step, record in test_pool.items():
        record['time_step'] = int(step)
        logsys.wandblog(record)
    for name, iters,values in iter_metric:
        iter_pick = np.linspace(0,len(iters)-1, len(epoch_pool)*3).astype('int')
        for step, val in zip(iters[iter_pick],values[iter_pick]):
            logsys.wandblog({'iter':step,name:val})
    logsys.close()
    print("all done..............")

class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

def run_fourcast(ckpt_path,step = 4*24//6,force_fourcast=False,wandb_id=None,weight_chose=None):
    #if "fourcastresult.gpu_0" in os.listdir(ckpt_path):return
    from train.pretrain import main
    args = get_the_args(ckpt_path)
    if args is None:return
    args.Train.mode      = 'fourcast'
    args.fourcast  = True
    args.recorder_list = []
    #args.use_wandb = 'off'
    args.use_offline_data = 2 if args.use_offline_data else 0
    args.SAVE_PATH = ckpt_path
    #args.Train.batch_size=args.Valid.valid_batch_size= 4 
    if wandb_id is not None:args.wandb_id = wandb_id
    if weight_chose is None:
        if 'backbone.best.pt' in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,'backbone.best.pt')
        elif 'pretrain_latest.pt' in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
        else:
            print(f"{ckpt_path}===>")
            print("   no backbone.best.pt or pretrain_latest.pt, pass!")
            return
    else:
        if weight_chose in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,weight_chose)
        else:
            print(f"{ckpt_path}===>")
            print("   no {weight_chose}, pass!")
            return
    #best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
    args.Checkpoint.pretrain_weight = best_path
    args.Dataset.time_step = step
    #args.data_root = "datasets/weatherbench"
    args.force_fourcast = force_fourcast
    
    if args.force_fourcast or 'rmse_table' not in os.listdir(ckpt_path):
        main(args)

def run_snap_nodal(ckpt_path,step = 4*24//6,force_fourcast=False,wandb_id=None,weight_chose=None):
    #if "fourcastresult.gpu_0" in os.listdir(ckpt_path) and not force_fourcast:return
    from train.pretrain import main
    args = get_the_args(ckpt_path)
    if args is None:return
    args.Train.mode      = 'fourcast_for_snap_nodal_loss' 
    args.fourcast  = True
    args.recorder_list = []
    #args.use_wandb = 'wandb_runtime'
    #args.Train.batch_size=args.Valid.valid_batch_size= 4 
    if wandb_id is not None:args.wandb_id = wandb_id
    if weight_chose is None:
        if 'backbone.best.pt' in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,'backbone.best.pt')
        elif 'pretrain_latest.pt' in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
        else:
            print(f"{ckpt_path}===>")
            print("   no backbone.best.pt or pretrain_latest.pt, pass!")
            return
    else:
        if weight_chose in os.listdir(ckpt_path):
            best_path = os.path.join(ckpt_path,weight_chose)
        else:
            print(f"{ckpt_path}===>")
            print("   no {weight_chose}, pass!")
            return
    #best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
    args.Checkpoint.pretrain_weight = best_path
    args.Dataset.time_step = step
    #args.data_root = "datasets/weatherbench"
    args.force_fourcast = force_fourcast
    args.wandb_id = None
    args.SAVE_PATH= ckpt_path
    main(args)

def get_the_args(ckpt_path):
    from train.pretrain import get_args
    if len(os.listdir(ckpt_path))==0:
        return 
    args = get_args(os.path.join(ckpt_path,'config.json'))
    args.SAVE_PATH = ckpt_path
    #     #args = Config(old_args)
    # else:
    #     if "Euler" in ckpt_path:
    #         args.wrapper_model   = [p for p in ckpt_path.split("/") if 'Euler' in p][0].split('-')[0]
    #args.Train_set = 'small'
    
    # args.time_reverse_flag = 'only_backward'
    # args.img_size  = (3,51,96)
    # args.patch_size= (1,3,3)
    return args

def remove_trail_path(trial_path):
    trail_file_list= os.listdir(trial_path)
    if ('seed' in trial_path.split('/')[-1] and
        'backbone.best.pt' not in trail_file_list and 
        'pretrain_latest.pt' not in trail_file_list and
        'accu_table' not in trail_file_list and
        'rmse_table' not in trail_file_list and
        'rmse_unit' not in trail_file_list):
        #os.system(f"ls {trail_path}")
        os.system(f"rm -rf {trail_path}")

def create_fourcast_table(ckpt_path,force_fourcast=False):
    if "figures" in os.listdir(ckpt_path) and not force_fourcast:return
    if "fourcastresult.gpu_0" not in os.listdir(ckpt_path):return
    from train.pretrain import create_fourcast_metric_table,get_test_dataset,LoggingSystem,parse_default_args,create_logsys
    import torch
    # if 'rmse_unit_table' in os.listdir(ckpt_path):
    #     return
    args = get_the_args(ckpt_path)
    
    args.Train.mode = 'fourcast'
    args.gpu = args.Pengine.engine.local_rank = gpu  = local_rank = 0
    #args.data_root = "datasets/weatherbench"
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    args.SAVE_PATH = ckpt_path
    ########## inital log ###################
    args.Pengine.engine.distributed = False 
    test_dataset,   test_dataloader = get_test_dataset(args)
    #args.SAVE_PATH = './debug'
    #args.use_wandb = 'wandb_runtime'
    logsys = create_logsys(args,False)

    info_pool_list = create_fourcast_metric_table(ckpt_path, logsys,test_dataset)
    dirname = summary_dir= ckpt_path
    dirname,name     = os.path.split(dirname)
    dirname,job_type = os.path.split(dirname)
    dirname,group    = os.path.split(dirname)
    dirname,project  = os.path.split(dirname)
    torch.save( info_pool_list, os.path.join(args.SAVE_PATH,f'{project}_{job_type}_{name}'))
    #torch.save( info_pool_list, os.path.join("debug",f'{project}_{job_type}_{name}'))
    for info_pool in info_pool_list:
        #print(info_pool)
        logsys.wandblog(info_pool)
    logsys.close()

def create_nodalsnap_table(ckpt_path):
    #if "figures" in os.listdir(ckpt_path):return
    if "nodal_snap_on_test_dataset.gpu_0" not in os.listdir(ckpt_path):return
    from train.pretrain import create_nodal_loss_snap_metric_table,get_test_dataset,LoggingSystem,parse_default_args,create_logsys
    import torch
    # if 'rmse_unit_table' in os.listdir(ckpt_path):
    #     return
    args = get_the_args(ckpt_path)
    args.Train.mode = 'fourcast_for_snap_nodal_loss'
    args.gpu = args.Pengine.engine.local_rank = gpu  = local_rank = 0
    #args.data_root = "datasets/weatherbench"
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    args.SAVE_PATH = ckpt_path
    ########## inital log ###################
    args.Pengine.engine.distributed = False
    test_dataset,   test_dataloader = get_test_dataset(args)
    #args.SAVE_PATH = './debug'
    args.use_wandb = 'wandb_runtime'
    args.wandb_id  = None
    logsys = create_logsys(args,False)
    info_pool_list = create_nodal_loss_snap_metric_table(ckpt_path, logsys,test_dataset)
    # dirname = summary_dir= ckpt_path
    # dirname,name     = os.path.split(dirname)
    # dirname,job_type = os.path.split(dirname)
    # dirname,group    = os.path.split(dirname)
    # dirname,project  = os.path.split(dirname)
    #torch.save( info_pool_list, os.path.join(args.SAVE_PATH,f'{project}_{job_type}_{name}'))
    #torch.save( info_pool_list, os.path.join("debug",f'{project}_{job_type}_{name}'))
    # for info_pool in info_pool_list:
    #     #print(info_pool)
    #     logsys.wandblog(info_pool)
    logsys.close()

def create_multi_fourcast_table(ckpt_path,force=False):
    from train.pretrain import create_multi_epoch_inference,get_test_dataset,parse_default_args,create_logsys
    if not np.any([("result_of_epoch" in t) for t in os.listdir(ckpt_path)]):return
    if not force and os.path.exists(os.path.join(ckpt_path,"result_of_epoch_0","fourcastresult.out")):
        return 
    args = get_the_args(ckpt_path)
    args.Train.mode = 'fourcast'
    args.gpu = args.Pengine.engine.local_rank = gpu  = local_rank = 0
    #args.data_root = "datasets/weatherbench"
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args= parse_default_args(args)
    args.SAVE_PATH = ckpt_path
    ########## inital log ###################
    args.Pengine.engine.distributed = False 
    test_dataset,   test_dataloader = get_test_dataset(args)
    logsys = create_logsys(args,False)
    result_path_list = [os.path.join(ckpt_path,result_path) for result_path in os.listdir(ckpt_path) if "result_of_epoch" in result_path]
    if len(result_path_list) == 0:return
    info_pool_list = create_multi_epoch_inference(result_path_list, logsys,test_dataset,force=force)
    logsys.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parse tf.event file to wandb', add_help=False)
    parser.add_argument('--paths',type=str,default="")
    parser.add_argument('--moded',type=str,default="dryrun")
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--divide', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--fourcast_step', default=4*24//6, type=int)
    parser.add_argument('--path_list_file',default="",type=str)
    parser.add_argument('--force_fourcast',default=0,type=int)
    parser.add_argument('--weight_chose',default=None,type=str)
    args = parser.parse_known_args()[0]

    if args.paths != "":
        level = args.level
        root_path = args.paths
        now_path = [root_path]
        while level>0:
            new_path = []
            for root_path in now_path:
                if os.path.isfile(root_path):continue
                if len(os.listdir(root_path))==0:
                    os.system(f"rm -r {root_path}")
                    continue
                for sub_name in os.listdir(root_path):
                    sub_path =  os.path.join(root_path,sub_name)
                    if os.path.isfile(sub_path):continue
                    new_path.append(sub_path)
            now_path = new_path
            level -= 1
    


    # now_path = [
    #     "checkpoints/WeathBench7066PatchDataset/POverLapTimePosBiasWrapper-AFNONet_Patch_7/time_step_3_finetune-2D70N_every_1_step/12_08_22_54_06-seed_2018",
    #     "checkpoints/WeathBench7066PatchDataset/POverLapTimePosBiasWrapper-AFNONet_Patch_7/time_step_4_pretrain-2D70N_every_1_step/12_08_22_57_30-seed_2018",
    #     "checkpoints/WeathBench7066PatchDataset/POverLapTimePosBiasWrapper-AFNONet_Patch_7/ts_3_finetune-2D70N_per_1_step_P9_for_P7/12_11_14_12_28-seed_2018",
    #     "checkpoints/WeathBench7066PatchDataset/POverLapTimePosBiasWrapper-AFNONet_Patch_7/ts_3_pretrain-2D70N_per_1_step_P9_for_P7/12_09_19_38_22-seed_2018  ",
    #     "checkpoints/WeathBench7066PatchDataset/POverLapTimePosBiasWrapper-AFNONet_Patch_7/ts_4_pretrain-2D70N_per_1_step_P10_for_P7/12_09_19_39_37-seed_2018"
    # ]
    now_path_pool = None
    if 'json' in args.path_list_file:
        with open(args.path_list_file, 'r') as f:
            path_list_file = json.load(f)
        if isinstance(path_list_file,dict):
            now_path_pool = path_list_file
            now_path = list(now_path_pool.keys())
        else:
            now_path = path_list_file

    # now_path = [
    #     "checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_49_50-seed_73001",
    #     "checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_50_07-seed_73001",
    #     "checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_50_11-seed_73001",
    #     "checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_50_15-seed_73001"
    # ]

    print(f"we detect {len(now_path)} trail path; \n  from {now_path[0]} to \n  {now_path[-1]}")
    total_lenght = len(now_path)
    length = int(np.ceil(1.0*total_lenght/args.divide))
    s    = int(args.part)
    now_path = now_path[s*length:(s+1)*length]
    print(f"we process:\n  from  from {now_path[0]}\n to  {now_path[-1]}")

    
    if args.moded == 'dryrun':exit()
    for trail_path in tqdm.tqdm(now_path):
        trail_path = trail_path.strip("/")
        if len(os.listdir(trail_path))==0:
            os.system(f"rm -r {trail_path}")
            continue
        #os.system(f"sensesync sync s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ {trail_path}/")
        #os.system(f"sensesync sync {trail_path}/ s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ ")
        #os.system(f"aws s3 --endpoint-url=http://10.140.2.204:80 --profile zhangtianning sync s3://FourCastNet/{trail_path}/ {trail_path}/")
        # print(trail_path)
        # print(os.listdir(trail_path))
        if   args.moded == 'fourcast':run_fourcast(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'tb2wandb':assign_trail_job(trail_path)
        elif args.moded == 'cleantmp':remove_trail_path(trail_path)
        elif args.moded == 'cleanwgt':remove_weight(trail_path)
        elif args.moded == 'createtb':create_fourcast_table(trail_path,force_fourcast=args.force_fourcast)
        elif args.moded == 'snap_nodal':run_snap_nodal(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'createtb_nodalsnap':create_nodalsnap_table(trail_path)
        elif args.moded == 'createmultitb':create_multi_fourcast_table(trail_path,force=args.force_fourcast)
        else:
            raise NotImplementedError
