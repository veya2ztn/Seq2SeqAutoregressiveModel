import argparse
import sys

def get_args_parser():
    
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--train_set', type=str, default='physics_small')

    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--valid_batch_size', default=-1, type=int)
    parser.add_argument('--epochs', default=-1, type=int)
    parser.add_argument('--save_warm_up', default=5, type=int)
    parser.add_argument('--more_epoch_train', default=0, type=int)
    parser.add_argument('--skip_first_valid', default=0, type=int)
    parser.add_argument('--epoch_save_list',default="99",type=str)
    parser.add_argument('--valid_every_epoch', default=1, type=int)
    parser.add_argument('--save_every_epoch', default=1, type=int)
    parser.add_argument('--tracemodel', default=0, type=int)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--input_noise_std', type=float, default=0.0, help='input_noise_std')
    parser.add_argument('--do_final_fourcast', type=int, default=1, help='do fourcast step after finish training')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--distributed', type=int, default=0, help='distributed')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--share_memory', type=int, default=1, help='share_memory_flag')
    parser.add_argument('--continue_train', type=int, default=0, help='continue_train')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='accumulation_steps')
    parser.add_argument('--use_wandb', type=str, default="off", help='when to activate wandb')
    parser.add_argument('--GDMod_type', type=str, default='off')
    parser.add_argument('--GDMod_lambda1', type=float, default=1)
    parser.add_argument('--GDMod_lambda2', type=float, default=0)
    parser.add_argument('--GDMod_L1_level', type=float, default=1)
    parser.add_argument('--GDMod_L2_level', type=float, default=1)
    parser.add_argument('--GDMod_sample_times', type=int, default=100)
    parser.add_argument('--path_length_mode', type=str, default="221")

    parser.add_argument('--GDMod_intervel',type=int,default=10)

    parser.add_argument('--path_length_regularize',type=int,default=0)
    parser.add_argument('--gmod_coef',type=str,default=None)

    parser.add_argument('--do_iter_log', type=int, default=1)
    parser.add_argument('--disable_progress_bar',type=int,default=0)
    parser.add_argument('--batch_limit', type=int, default=1)
    parser.add_argument('--split', type=str, default="")
    parser.add_argument('--log_trace_times', type=int, default=None)
    
    # Model parameters
    parser.add_argument('--model_type', default='AFNONet', type=str, help='Name of model to train',
                        #choices=['AFNONet','FEDformer','FEDformer1D','AFNONetJC','NaiveConvModel2D']
                        )
    parser.add_argument('--patch_size', default="", type=str)
    parser.add_argument('--img_size'  , default="", type=str)
    parser.add_argument('--modes'  , default="17,33,6", type=str)
    parser.add_argument('--mode_select', default="normal", type=str)
    parser.add_argument('--physics_num' , type=int, default=4)
    parser.add_argument('--input_channel' , type=int, default=0)
    parser.add_argument('--output_channel', type=int, default=0)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--model_depth', type=int, default=12)
    parser.add_argument('--history_length', default=1, type=int)
    parser.add_argument('--block_target_timestamp', default=0, type=int)
    parser.add_argument('--canonical_fft', default=1, type=int)
    parser.add_argument('--unique_up_sample_channel', default=0, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--label_len', type=int, default=3)
    parser.add_argument('--use_amp', type=int, default=1, help='use_amp')
    parser.add_argument('--random_time_step', action='store_true')
    parser.set_defaults(random_time_step=False)
    parser.add_argument('--use_scalar_advection', action='store_true')
    parser.set_defaults(use_scalar_advection=False)
    parser.add_argument('--patch_range', type=str, default=5, help='patch_range')
    parser.add_argument('--dataset_patch_range', type=str, default=None, help='dataset_patch_range')
    parser.add_argument('--criterion', type=str, default='mse', help='criterion')
    parser.add_argument('--ngmod_freq', type=int, default=10, help='ngmod_freq')
    parser.add_argument('--split_batch_chunk', type=int, default=16, help='split_batch_chunk')
    parser.add_argument('--gmod_update_mode', type=int, default=2, help='gmod_update_mode')
    
    #### fno parameters
    parser.add_argument('--fno-bias', action='store_true')
    parser.add_argument('--fno-blocks', type=int, default=4)
    parser.add_argument('--fno-softshrink', type=float, default=0.00)
    parser.add_argument('--double-skip', action='store_true')
    parser.add_argument('--tensorboard-dir', type=str, default=None)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--checkpoint-activations', action='store_true')
    parser.add_argument('--autoresume', action='store_true')
    
    parser.add_argument('--wrapper_model', default='', type=str, help='Name of model to train',
        #   choices=["",'DeltaModel','EulerEquationModel','ConVectionModel',
        #         'EulerEquationModel3','Time_Sphere_Model','EulerEquationModel2',
        #         'EulerEquationModel4','OnlineNormModel','Time_Projection_Model','DirectSpace_Feature_Model']
                )

    # Dataset parameters
    parser.add_argument('--dataset_type', default='', type=str, help='Name of dataset to train',choices=["",
    'WeathBench71','ERA5Tiny12_47_96_Normal','ERA5CephDataset','WeathBench68pixelnorm','WeathBench69SolarLunaMask',
    'WeathBench7066deseasonal','WeathBench7066PatchDataset','ERA5CephSmallPatchDataset',
    'WeathBench7066Self','ERA5CephSmallDataset','ERA5Tiny12_47_96','WeathBench7066',
    'WeathBench7066DeltaDataset','WeathBench55withoutH'])
    parser.add_argument('--dataset_flag', default="", type=str)
    parser.add_argument('--time_reverse_flag', default='only_forward', type=str)
    parser.add_argument('--time_intervel', type=int, default=1)
    parser.add_argument('--time_step', type=int, default=-1)
    parser.add_argument('--data_root', type=str, default="")
    parser.add_argument('--use_time_stamp', type=int, default=0)
    parser.add_argument('--use_position_idx', type=int, default=0)
    parser.add_argument('--cross_sample', type=int, default=1)
    parser.add_argument('--use_inmemory_dataset',type=int,default=0)
    parser.add_argument('--random_dataset',type=int,default=0)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--use_offline_data',type=int,default=0)
    parser.add_argument('--chunk_size',type=int,default=1024)
    parser.add_argument('--picked_inputoutput_property',type=str,default=None)
    
    # Fourcast Parameter
    parser.add_argument('--pretrain_weight', type=str, default='', help='pretrain_weight')
    parser.add_argument('--fourcast_randn_initial', default=0, type=int)
    parser.add_argument('--force_fourcast', default=0, type=int)
    parser.add_argument('--snap_index', type=str, default=None)
    parser.add_argument('--wandb_id', type=str, default=None)
    
    # Optimizer parameters # feed into timm
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=-1, metavar='LR', help='learning rate (default: 5e-4)')

    # scheduler parameters # feed into timm
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')



    # Augmentation parameters
    # parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    # parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". "(default: rand-m9-mstd0.5-inc1)'),
    # parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    # parser.add_argument('--train-interpolation', type=str, default='bicubic',  help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # parser.add_argument('--repeated-aug', action='store_true')
    # parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    # parser.add_argument('--reprob', type=float, default=0, metavar='PCT', help='Random erase prob (default: 0.25)')
    # parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    # parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')


    # long short parameters
    # parser.add_argument('--ls-w', type=int, default=4)
    # parser.add_argument('--ls-dp-rank', type=int, default=16)

    return parser

import json
def get_args(config_path=None):
    conf_parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,add_help=False)
    conf_parser.add_argument("-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {}
    if config_path:
        with open(config_path,'r') as f:
            defaults = json.load(f)
    if args.conf_file:
        with open(args.conf_file,'r') as f:
            defaults = json.load(f)
    #parser = argparse.ArgumentParser(parents=[conf_parser])
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    parser.set_defaults(**defaults)

    config = parser.parse_known_args(remaining_argv)[0]
    config.config_file = args.conf_file
    return config

if __name__ == '__main__':
    args = (get_args())
    for key, val in vars(args).items():
        print(f"{key:30s} ---> {val}")