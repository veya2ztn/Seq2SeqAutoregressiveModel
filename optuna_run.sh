#!/bin/sh
#SBATCH -J F7066A3      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-F7066A3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-F7066A3.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

# python optuna_run.py --train_set 4796ad --epoch 60 --mode pretrain --time_step 2 --patch_size 1,3,3 \
#                      --img_size 3,51,96 --dataset_type ERA5Tiny12_47_96 --input_channel 4 --output_channel 4 \
#                       --use_scalar_advection
#python optuna_run.py --train_set 3D70N --epoch 60 --mode pretrain --time_step 8  --history_length 6 --model_depth 6 --embed_dim 256 --patch_size 2
#python optuna_run.py --train_set 2D706N --epoch 300 --mode pretrain --time_step 8  --history_length 6

#--img_size 3,32,64 --patch_size 1,2,2
#--activate_physics_model

# python optuna_run.py --train_set 47_96_normal --epoch 60 --mode pretrain --time_step 3 --patch_size 1,3,3 \
#                      --img_size 3,51,96 --dataset_type ERA5Tiny12_47_96_Normal --input_channel 4 --output_channel 4 \
#                       --wrapper_model DeltaModel

#python optuna_run.py --train_set physics_small --epoch 100 --mode pretrain --time_step 3  --time_reverse_flag random_forward_backward
# python optuna_run.py --train_set small --use_time_stamp 1  --history_length 6 --time_step 7 \
# --model_type FEDformer --embed_dim 256 --use_amp 0 --batch_size 16 --model_depth 1 --valid_batch_size 32 \
# --clip-grad 1e4


####### ----------------> AFNONetJC-WeathBench7066 <--------------------------
#python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --model_type AFNONetJC --time_step 2 --epoch 100 --mode pretrain --dataset_flag 3D70U  --wrapper_model ConVectionModel --debug 0 --use_amp 0
#python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --model_type AFNONetJC --time_step 2 --epoch 100 --mode pretrain  --debug 0 --use_amp 0


####### ----------------> AFNONet-WeathBench7066 <--------------------------
# python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --time_step 3 --epoch 100 --mode pretrain \
# --dataset_flag 3D70U  --wrapper_model ConVectionModel --use_amp 0 --patch_size 2 
# python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --time_step 3 --epoch 40 --mode finetune \
# --dataset_flag 3D70U  --wrapper_model ConVectionModel --use_amp 0 --patch_size 2 \
# --pretrain_weight checkpoints/WeathBench7066/ConVectionModel-AFNONet/time_step_2_pretrain-2D706N_every_1_step/10_07_00_19_40-seed_2641/backbone.best.pt 

#python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --time_step 2 --epoch 100 --mode pretrain  --patch_size 2 
python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --time_step 3 --epoch 100 --mode finetune --patch_size 2  \
--pretrain_weight checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/10_08_21_16_55-seed_28356/backbone.best.pt

python optuna_run.py --train_set 2D706N --dataset_type WeathBench7066 --time_step 2 --epoch 40 --mode pretrain \
--dataset_flag 3D70U  --wrapper_model DirectSpace_Feature_Model --use_amp 0 --patch_size 2 