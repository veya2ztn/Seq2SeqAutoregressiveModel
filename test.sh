export CUDA_VISIBLE_DEVICES=4,5,6,7;export THEPATH=checkpoints/WeathBench7066/AFNONet/ts_2_pretrain-2D706N_per_1_step/01_06_18_50-seed_73001;\
python train/pretrain.py -c $THEPATH/config.json --batch_size 16 --time_step 3 --mode finetune --pretrain_weight $THEPATH/backbone.best.pt \
--compute_graph_set fwd2_D_Mog --do_iter_log 2 --save_warm_up 2 --accumulation_steps 1 --use_wandb wandb_runtime
python train/pretrain.py -c $THEPATH/config.json --batch_size 16 --time_step 4 --mode finetune --pretrain_weight $THEPATH/backbone.best.pt \
--compute_graph_set fwd3_D_Mog --do_iter_log 2 --save_warm_up 2 --accumulation_steps 1 --use_wandb wandb_runtime
python train/pretrain.py -c $THEPATH/config.json --batch_size 16 --time_step 4 --mode finetune --pretrain_weight $THEPATH/backbone.best.pt \
--compute_graph_set fwd3_D_Rog5 --do_iter_log 2 --save_warm_up 2 --accumulation_steps 1 --use_wandb wandb_runtime
