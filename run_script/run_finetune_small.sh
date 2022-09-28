#!/bin/sh
#SBATCH -J fst-ft-small      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o fourcast-fintune_small.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e fourcast-fintune_small.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --mode finetune --pretrain_weight checkpoints/fourcastnet/pretrain-small/08_01_15_34/backbone.best.pt --train_set small
