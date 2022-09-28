#!/bin/sh
#SBATCH -J 2D706NT8      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-2D706NT8.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-2D706NT8.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set 2D706N --epoch 80 --mode pretrain --time_step 7 --history_length 6 --time_intervel 8 --model_depth 6