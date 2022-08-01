#!/bin/sh
#SBATCH -J fourcast-fintune      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o fourcast-fintune.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e fourcast-fintune.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --mode finetune --pretrain_weight checkpoints/fourcastnet/pretrain/07_23_19_35_04/pretrain_latest.pt
