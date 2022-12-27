#!/bin/sh
#SBATCH -J HFF-adam      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-HFF-adam.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-HFF-adam.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python train/pretrain.py --train_set small --use_time_stamp 1  --history_length 6 --time_step 7 --model_type FEDformer \
--embed_dim 128 --use_amp 0 --batch_size 16 --model_depth 2 --valid_batch_size 32 \
--clip-grad 1e2 --label_len 4 --lr 0.001 --opt adam --modes 32,33,6 --sched "" --n_heads 256 --depth 10 --share_memory 0