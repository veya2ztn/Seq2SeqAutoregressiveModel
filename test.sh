for dataset_flag in  3D70N #3D70N
do
for flag in train #test valid 
do
    nohup python test.py $dataset_flag $flag > log/offlinedata.$dataset_flag.$flag.2&
done
done

export CUDA_VISIBLE_DEVICES=5
export CKPTPATH=checkpoints/WeathBench7066/AFNONet/time_step_2_pretrain-2D706N_every_1_step/11_21_20_50_07-seed_73001
nohup python -u train/pretrain.py -c $CKPTPATH/config.json --pretrain_weight $CKPTPATH/pretrain_latest.pt \
--GDMod_type NGmod_estimate_L2 --GDMod_lambda1 1 --GDMod_lambda2 1 --GDMod_sample_times 18 \
--accumulation_steps 4 --batch_size 4 --valid_batch_size 16 > log/L2train.task$CUDA_VISIBLE_DEVICES.log&