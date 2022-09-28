#!/bin/sh
#SBATCH -J fst-pre-s      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o fourcast-pretrain-small3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e fourcast-pretrain-small3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set small --epoch 100 --seed 1 --batch-size 32
