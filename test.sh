# for dataset_flag in  3D70N #3D70N
# do
# for flag in train #test valid 
# do
#     nohup python test.py $dataset_flag $flag > log/offlinedata.$dataset_flag.$flag.2&
# done
# done

# export CUDA_VISIBLE_DEVICES=5
# export CKPTPATH=checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_50_07-seed_73001
# nohup python -u train/pretrain.py -c $CKPTPATH/config.json --pretrain_weight $CKPTPATH/pretrain_latest.pt \
# --GDMod_type NGmod_estimate_L2 --GDMod_lambda1 1 --GDMod_lambda2 1 --GDMod_sample_times 18 \
# --accumulation_steps 4 --batch_size 4 --valid_batch_size 16 > log/L2train.task$CUDA_VISIBLE_DEVICES.log&

export THEPATH=checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_06_18_50-seed_73001;\
export CUDA_VISIBLE_DEVICES=0,1,2,3;\
python -u train/pretrain.py -c $THEPATH/config.json --dataset_type WeathBench68pixelnorm --dataset_flag 2D68K --input_channel 68 \
--output_channel 68 --use_offline_data 2 --fourcast_during_train 5  --do_error_propagration_monitor 1

export THEPATH=checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_06_18_50-seed_73001;\
export CUDA_VISIBLE_DEVICES=0,1,2,3;\
python -u train/pretrain.py -c $THEPATH/config.json --dataset_type WeathBench68pixelnorm --dataset_flag 2D68K --input_channel 68 \
--output_channel 68 --use_offline_data 2 --fourcast_during_train 5  --do_error_propagration_monitor 1 --batch_size 4

export THEPATH=checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_06_18_50-seed_73001;\
export CUDA_VISIBLE_DEVICES=0,1,2,3;\
python -u train/pretrain.py -c $THEPATH/config.json --dataset_type WeathBench68pixelnorm --dataset_flag 2D68K --input_channel 68 \
--output_channel 68 --use_offline_data 2 --fourcast_during_train 5  --do_error_propagration_monitor 1 --lr 0.001
