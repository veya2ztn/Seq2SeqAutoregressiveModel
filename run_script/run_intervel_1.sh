#!/bin/sh

python train/pretrain.py --train_set $1 --epoch 80 --mode pretrain --time_step 7 --history_length 6 --time_intervel $2 --model_depth 6 --pretrain_weight checkpoints/WeathBench71/small_AFNONet/history_6_time_step_7_pretrain-$1_every_$2_step/resume/pretrain_latest.pt