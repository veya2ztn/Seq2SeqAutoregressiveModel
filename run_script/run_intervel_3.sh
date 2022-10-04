#!/bin/sh
#SBATCH -J F2D70NT3      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-F2D70NT3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-F2D70NT3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set 2D70N --epoch 50 --mode finetune --time_step 8 --history_length 6 --time_intervel 3 \
--model_depth 6 --pretrain_weight checkpoints/WeathBench71/small_AFNONet/history_6_time_step_8_finetune-2D70N_every_3_step/10_03_11_51-seed_27468/pretrain_latest.pt --seed 42 \
--continue_train 1 \
--data_root 'weatherbench:s3://weatherbench/weatherbench32x64/npy'