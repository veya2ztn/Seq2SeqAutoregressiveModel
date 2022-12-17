import os
import re
import json
epoch_patter = re.compile(r'epoch:(\d+)')   # 查找数字
iter_patter  = re.compile(r'iter:(.*?)/')
itert_patter = re.compile(r'/\[(.*?)\]')
abs_loss_pat = re.compile(r'abs_loss:(.*) loss')
loss_pat     = re.compile(r' loss:(.*) cost')

epoch_finder = lambda string:int(epoch_patter.findall(string)[0])
iter_finder  = lambda string:int(iter_patter.findall(string)[0].strip(']').strip('['))
itert_finder = lambda string:int(itert_patter.findall(string)[0].strip(']').strip('['))
abs_ls_finder= lambda string:float(abs_loss_pat.findall(string)[0])
loss_finder  = lambda string:float(loss_pat.findall(string)[0])

import sys
from utils.params import get_args
args = get_args()
with open("checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/10_14_20_18_11-seed_73001/config.json", 'r') as f:old_args = json.load(f)

for key,val in old_args.items():
    if hasattr(args,key):
        setattr(args,key,val)
args.GDMod_type = sys.argv[1]#'NGmod_absolute'
args.accumulation_steps = int(sys.argv[2])#1
args.batch_size = int(sys.argv[3])#4
log_name = sys.argv[4]#'GPU1.A1.B16.NGmod_absolute'
print(args.GDMod_type)
print(args.accumulation_steps)
print(args.batch_size)
print(log_name)

args.use_wandb = 'wandb_runtime'
from train.pretrain import create_logsys,parse_default_args,get_ckpt_path
args= parse_default_args(args)
SAVE_PATH = get_ckpt_path(args)
args.SAVE_PATH  = str(SAVE_PATH)
args.gpu=0
logsys = create_logsys(args)
tables = []
with open(f"old_log/{log_name}","r") as f:
    for line in f:
        if "│" in line:
            if 'epoch:' in line:
                epoch_now = epoch_finder(line)
                _, epoch_block, valid_block, train_block,_ = line.split("│")
                valid = float(valid_block)
                train = float(train_block)
                if valid == -1:continue
                tables.append([epoch_now,valid,train])
                #print(f"epcoh:{epoch_now} valid:{valid} train:{train}")
import pandas as pd
table = pd.DataFrame(tables,columns=["epoch","valid_loss","train_loss"])
table.to_csv(os.path.join(SAVE_PATH,'recovery_loss_table'))
for epoch, valid, train in tables:
    logsys.wandblog({"epoch":epoch,"valid":valid,"train":train})
logsys.close()

