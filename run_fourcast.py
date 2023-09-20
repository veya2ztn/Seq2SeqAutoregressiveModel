from genericpath import isfile
import pwd,os
import sys
import numpy as np
import shutil
from train.pretrain import *
# datadir = "/home/PJLAB/zhangtianning/Documents/dataset/ERA5"
# for feature_name in os.listdir(datadir):
#     feature_file_path = os.path.join(datadir, feature_name)
#     if os.path.isfile(feature_file_path):continue
#     for file_name in os.listdir(feature_file_path):
#         full_file_path = os.path.join(feature_file_path,file_name)
#         if not os.path.isfile(full_file_path):continue
#         source_path = full_file_path
#         for year in ["2016","2017","2018"]:
#             target_dir  = os.path.join(feature_file_path,year)
#             if not os.path.exists(target_dir):os.makedirs(target_dir)
#             if year in file_name:
#                 target_path = os.path.join(target_dir,file_name)
#                 os.system(f"mv {source_path} {target_path}")
#                 #shutil.move(source_path,target_path)

def run_fourcast(ckpt_path):
    args = get_args(args=[])
    args.Train_set = "physics_small"
    args.fourcast  = True
    args.Train.mode = 'fourcast'
    args.Train.batch_size= 64
    if 'backbone.best.pt' in os.listdir(ckpt_path):
        best_path = os.path.join(ckpt_path,'backbone.best.pt')
    elif 'pretrain_latest.pt' in os.listdir(ckpt_path):
        best_path = os.path.join(ckpt_path,'pretrain_latest.pt')
    else:
        return 
    if 'rmse_table_unit' in os.listdir(ckpt_path):
        return 
    args.Checkpoint.pretrain_weight = best_path
    if 'config.json' in os.listdir(ckpt_path):
        with open(os.path.join(ckpt_path,'config.json'),'r') as f:
            old_args = json.load(f)
        args.patch_size      = old_args['patch_size']
    if "Euler" in ckpt_path:
        args.wrapper_model   = [p for p in ckpt_path.split("/") if 'Euler' in p][0].split('-')[0]
    main(args)

import sys 
import numpy as np
fourcast_ckpt_list=[
        "checkpoints/ERA5_20-12/EulerEquationModel2-AFNONet/time_step_6_pretrain-physics_small/08_26_14_20_26-seed_42",
        "checkpoints/ERA5_20-12/EulerEquationModel-AFNONet/pretrain-physics_small/08_23_07_40_16-seed_52831",
        "checkpoints/ERA5_20-12/EulerEquationModel-AFNONet/pretrain-physics_small/08_23_07_19_28-seed_51025",
        "checkpoints/ERA5_20-12/EulerEquationModel-AFNONet/pretrain-physics_small/08_23_02_11_19-seed_89651",
    ]

total_num = len(fourcast_ckpt_list)
split = 1
range_list = np.linspace(0,total_num,split+1).astype('int')
num = int(sys.argv[1])
start, end = range_list[num], range_list[num+1]

for idx in range(start,end):
    ckpt_path = fourcast_ckpt_list[idx]
    run_fourcast(ckpt_path)
