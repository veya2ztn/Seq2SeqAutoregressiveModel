#!/bin/sh
#SBATCH -J HalfFFer      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-HalfFFer.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-HalfFFer.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

# python optuna_run.py --train_set 4796ad --epoch 60 --mode pretrain --time_step 2 --patch_size 1,3,3 \
#                      --img_size 3,51,96 --dataset_type ERA5Tiny12_47_96 --input_channel 4 --output_channel 4 \
#                      --save_warm_up 1000 --use_scalar_advection
#python optuna_run.py --train_set 3D70N --epoch 60 --mode pretrain --time_step 8 --save_warm_up 1000 --history_length 6 --model_depth 6 --embed_dim 256 --patch_size 2
#python optuna_run.py --train_set 2D706N --epoch 300 --mode pretrain --time_step 8 --save_warm_up 1000 --history_length 6

#--img_size 3,32,64 --patch_size 1,2,2
#--activate_physics_model

# python optuna_run.py --train_set 47_96_normal --epoch 60 --mode pretrain --time_step 3 --patch_size 1,3,3 \
#                      --img_size 3,51,96 --dataset_type ERA5Tiny12_47_96_Normal --input_channel 4 --output_channel 4 \
#                      --save_warm_up 1000 --wrapper_model DeltaModel

#python optuna_run.py --train_set physics_small --epoch 100 --mode pretrain --time_step 3 --save_warm_up 1000 --time_reverse_flag random_forward_backward
python train/pretrain.py --train_set small --use_time_stamp 1  --history_length 6 --time_step 7 \
--model_type FEDformer --embed_dim 256 --use_amp 0 --batch_size 16 --model_depth 1 --valid_batch_size 32 \
--clip-grad 1e4