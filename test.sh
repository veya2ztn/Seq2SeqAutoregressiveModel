for THEPATH in \
checkpoints/WeathBench7066/AFNONet/ts_3_finetune-2D706N_per_1_step/01_26_17_38_64254-seed_73001 \
checkpoints/WeathBench7066/AFNONet/ts_3_pretrain-2D706N_per_1_step/01_26_23_35_64256-seed_73001 \
checkpoints/WeathBench7066/AFNONet/ts_4_finetune-2D706N_per_1_step/01_26_23_56_64255-seed_73001 \
checkpoints/WeathBench7066/AFNONet/ts_4_pretrain-2D706N_per_1_step/01_26_17_38_64248-seed_73001
do
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;python mytool.py --level 0 --mode fourcast --do_error_propagration_monitor 8 --use_wandb off --force_fourcast 1 --fourcast_step 22 --paths $THEPATH
done 
#checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_26_14_21_64252-seed_73001 \