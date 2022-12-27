#!/bin/sh
#SBATCH -J FullFFer      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-FullFFer.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-FullFFer.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set small --use_time_stamp 1  --history_length 6 --time_step 7 \
--model_type FEDformer --embed_dim 256 --use_amp 0 --batch_size 4 --model_depth 2 --modes 32,64,6 \
--valid_batch_size 12 --label_len 4 --lr 0.001 --sched ""