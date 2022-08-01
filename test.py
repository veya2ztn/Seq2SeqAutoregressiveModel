from genericpath import isfile
import pwd,os
import sys
import numpy as np
import shutil
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
datadir = "/mnt/lustre/zhangtianning/projects/FourCastNet/output/fourcastnet/"
for  trail in os.listdir(datadir):
    trail_path = os.path.join(datadir,trail)
    if len(os.listdir(trail_path))==0:
        os.system(f"rm -rf {trail_path}")
        pass
    else:
        print(trail_path)