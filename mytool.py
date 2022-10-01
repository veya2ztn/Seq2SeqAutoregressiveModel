from ast import Not
import os
os.environ['WANDB_SILENT']="true"

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

def assign_trail_job(trial_path):
    from tbparse import SummaryReader
    import wandb
    dirname = summary_dir= trial_path
    dirname,name     = os.path.split(dirname)
    dirname,job_type = os.path.split(dirname)
    dirname,group    = os.path.split(dirname)
    dirname,project  = os.path.split(dirname)
    # if 'rmse_table_unit' not in os.listdir(trial_path):
    #     return
    job_type = job_type.replace("random_step_","")

    epoch_pool  = {}
    test_pool   = {}
    iter_metric = []
    hparams1=hparams2=None
    mode      =  'pretrain' if 'pretrain' in trial_path else 'finetune'
    pattern   = re.compile(f'time_step_(.*)_{mode}')   # 查找数字
    time_step = pattern.findall(trial_path)
    if len(time_step)==0:
        time_step=2
    else:
        time_step=int(time_step[0])

    for filename in os.listdir(summary_dir):
        if 'event' not in filename:continue
        log_dir = os.path.join(summary_dir,filename)
        reader = SummaryReader(log_dir)
        df = reader.scalars
        if 'tag' not in reader.hparams:
            print(trial_path)
            print(os.listdir(trial_path))
            print(reader.hparams)
          
        if len(df) < 1: continue

        for key in set(df['tag'].values):
            all_pool = test_pool if 'test' in key else epoch_pool
            if len(df[df['tag'] == key] )>1e3:
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
    hparams = hparams2 if hparams1 is None else hparams1
    if hparams == None:return
    hparams['mode']      = mode
    hparams['time_step'] = time_step
    wandb.init(config  = hparams,
            project = project,
            entity  = "szztn951357",
            group   = group,
            job_type= job_type,
            name    = name,
            #dir     = "ERA5_20-12/AFNONet/pretrain-physics_small/08_11_21_34",
            settings=wandb.Settings(_disable_stats=True)
            #reinit=True
            )
    for step, record in epoch_pool.items():
        record['epoch'] = int(step)
        wandb.log(record)
    for step, record in test_pool.items():
        record['time_step'] = int(step)
        wandb.log(record)
    for name, iters,values in iter_metric:
        iter_pick = np.linspace(0,len(iters)-1, len(epoch_pool)*3).astype('int')
        for step, val in zip(iters[iter_pick],values[iter_pick]):
            wandb.log({'iter':step,name:val})
    wandb.finish()

class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

def run_fourcast(ckpt_path):
    from train.pretrain import main,get_args

    args = get_args(args=[])
        
    
    if 'backbone.best.pt' in os.listdir(ckpt_path):
        best_path = os.path.join(ckpt_path,'backbone.best.pt')
    elif 'pretrain_latest.pt' in os.listdir(ckpt_path):
        best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
    else:
        print("no backbone.best.pt or pretrain_latest.pt, pass!")
        return
    if 'rmse_unit_table' in os.listdir(ckpt_path):
        return
    #args.train_set = "small"
    args.fourcast  = True
    args.mode      = 'fourcast'
    args.batch_size= 128
    args.pretrain_weight = best_path

    if 'config.json' in os.listdir(ckpt_path):
        with open(os.path.join(ckpt_path,'config.json'),'r') as f:
            old_args = json.load(f)
        for key,val in old_args.items():
            if hasattr(args,key):
                setattr(args,key,val)
        #args = Config(old_args)
    else:
        if "Euler" in ckpt_path:
            args.wrapper_model   = [p for p in ckpt_path.split("/") if 'Euler' in p][0].split('-')[0]
    #args.train_set = 'small'
    args.mode      = 'fourcast'
    args.fourcast  = True
    args.batch_size= 128    
    args.pretrain_weight = best_path
    args.time_step = 4*24//6
    # args.time_reverse_flag = 'only_backward'
    # args.img_size  = (3,51,96)
    # args.patch_size= (1,3,3)

    main(args)

def remove_trail_path(trial_path):
    trail_file_list= os.listdir(trial_path)
    if ('seed' in trial_path.split('/')[-1] and
        'backbone.best.pt' not in trail_file_list and 
        'pretrain_latest.pt' not in trail_file_list and
        'accu_table' not in trail_file_list and
        'rmse_table' not in trail_file_list and
        'rmse_unit' not in trail_file_list):
        os.system(f"rm -rf {trail_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser('parse tf.event file to wandb', add_help=False)
    parser.add_argument('--path',type=str)
    parser.add_argument('--mode',type=str)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--divide', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    args = parser.parse_args()

    level = args.level
    root_path = args.path
    now_path = [root_path]
    while level>0:
        new_path = []
        for root_path in now_path:
            for sub_name in os.listdir(root_path):
                sub_path =  os.path.join(root_path,sub_name)
                if os.path.isfile(sub_path):continue
                new_path.append(sub_path)
        now_path = new_path
        level -= 1
    total_lenght = len(now_path)
    length = int(np.ceil(1.0*total_lenght/args.divide))
    s = args.part

    print(f"we detect {len(now_path)} trail path; from {now_path[0]} to {now_path[-1]}")
    now_path = now_path[s*length:(s+1)*length]
    print(f"we process from {now_path[0]} to {now_path[-1]}")

    # now_path = [
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_2_pretrain-physics_small/08_25_20_21_40-seed_42",        
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_3_pretrain-physics_small/09_16_22_52_10-seed_47117",
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_4_pretrain-physics_small/09_15_04_24_58-seed_36823",
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_5_pretrain-physics_small/09_15_08_38_29-seed_91367",
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_6_pretrain-physics_small/09_17_23_28_33-seed_27848",
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_7_pretrain-physics_small/09_17_13_25_48-seed_79912",
    #     "checkpoints/ERA5_20-12/AFNONet/time_step_8_pretrain-physics_small/09_16_18_53_07-seed_69917"

    # ]

    for trail_path in tqdm.tqdm(now_path):
        #os.system(f"sensesync sync s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ {trail_path}/")
        #os.system(f"sensesync sync {trail_path}/ s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ ")
        #os.system(f"aws s3 --endpoint-url=http://10.140.2.204:80 --profile zhangtianning sync s3://FourCastNet/{trail_path}/ {trail_path}/")
        # print(trail_path)
        # print(os.listdir(trail_path))

        if   args.mode == 'fourcast':run_fourcast(trail_path)
        elif args.mode == 'tb2wandb':assign_trail_job(trail_path)
        elif args.mode == 'cleantmp':remove_trail_path(trail_path)
        elif args.mode == 'cleanwgt':remove_weight(trail_path)
        else:
            raise NotImplementedError