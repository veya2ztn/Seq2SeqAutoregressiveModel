#!/bin/sh
#SBATCH -J fourcast-pretrain      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/fourcast-pretrain.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/fourcast-pretrain.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set physics --seed 1 --activate_physics_model

