import argparse
import json
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
from model.model_arguments import (AFNONetConfig, PatchEmbeddingConfig, GraphCastConfig)

def build_parser():
    parser = ArgumentParser(description='Arguments', allow_abbrev=False, add_help=True)
    # Standard arguments.
    parser = _add_model_args(parser)
    parser = _add_training_args(parser)
    parser = _add_optimizer_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_valid_args(parser)
    return parser

def get_plain_parser():
    parser = build_parser()
    return parser.parse_args()

def structure_args(args):
    
    new_args = argparse.Namespace(
        model=get_model_args(args),
        train=get_train_args(args),
        valid=get_valid_args(args),
        data=get_data_args(args),
        optimizer=get_optim_args(args),
        debug=args.debug
    )
    return new_args

def get_args_parser():
    """Parse all arguments in structure way"""
    args = get_plain_parser()
    args = structure_args(args)   
    return args

def flatten_dict(_dict):
    out = {}
    for key, val in _dict.items():
        if isinstance(val, dict):
            for k, v in val.items():
                out[k] = v 
        else:
            out[key] = val
    return out
    
def get_args(config_path=None):
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--conf_file",     default=None, help="Specify config file", metavar="FILE")
    conf_parser.add_argument("-m", "--model_config",  default=None, help="Specify config file", metavar="FILE")
    conf_parser.add_argument("-t", "--train_config",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser('The Whole argument', parents=[build_parser()])
    defaults = {}
    config_path = config_path if config_path else args.conf_file

    if config_path:
        with open(config_path, 'r') as f:defaults = json.load(f)
        
        if 'model_config' in defaults:
            args.Model.model_config = defaults['model_config']
            print("given config has model config, overwrite other model_config")
        if 'train_config' in defaults:
            args.Train_config = defaults['train_config']
            print("given config has train config, overwrite other train_config")
        new_pool = {}
        for key, val in defaults.items(): 
            # check the code below , there are many key and attr have different name, which means the json load cannot recovery exactly the origin configuration.
            # since the model_config and train_config maybe also get changed during develop.
            # load a trail from json config is experiment.
            if isinstance(val, dict):
                for k,v in val.items():new_pool[k] = v
            else:
                new_pool[key] = val
        parser.set_defaults(**flatten_dict(new_pool))

    if args.Model.model_config:
        with open(args.Model.model_config, 'r') as f:defaults = json.load(f)
        parser.set_defaults(**flatten_dict(defaults))

    if args.Train_config:
        with open(args.Train_config, 'r') as f:defaults = json.load(f)
        parser.set_defaults(**flatten_dict(defaults)) 
    
    
    config = parser.parse_known_args(remaining_argv)[0]
    config.config_file = args.conf_file
    config = structure_args(config)

    if args.Model.model_config:config.model_config = args.Model.model_config
    if args.Train_config:config.train_config = args.Train_config
    return config



##############################################
############### Model Setting ################
##############################################

def _add_model_args(parser):
    group = parser.add_argument_group(title='model')
    
    group.add_argument('--history_length', type=int, default=1)
    group.add_argument('--in_chans', type=int,       default=20)
    group.add_argument('--out_chans', type=int,      default=20)
    group.add_argument('--embed_dim', type=int,      default=768)
    group.add_argument('--depth', type=int,          default=12)
    group.add_argument('--num_heads', type=int,      default=16)
    group.add_argument('--compute_graph_set'  , type=str, default=None)


    group.add_argument('--skip_constant_2D70N', action='store_true',default=False)
    group.add_argument('--pos_embed_type'     , type=str,default=None)
    group.add_argument('--patch_size', type=int, nargs=',' , default=8)

    parser.add_arguments(AFNONetConfig, dest="afnonet")
    #parser.add_arguments(PatchEmbeddingConfig, dest="patch_embedding")
    parser.add_arguments(GraphCastConfig, dest="graphcast")
    # ### AFNONET
    # group.add_argument('--unique_up_sample_channel', type=int, default=None)
    # group.add_argument('--fno-softshrink', type=float, default=0.00)
    # group.add_argument('--fno-bias', action='store_true')
    # group.add_argument('--double-skip', action='store_true')

    # #### Sphere Model
    # group.add_argument('--block_target_timestamp', default=0, type=int)
    
    # #### MultiBranch Model
    # group.add_argument('--model_type1', type=str, default=None)
    # group.add_argument('--model_type2', type=str, default=None)
    # group.add_argument('--backbone1_ckpt_path', type=str, default="")
    # group.add_argument('--backbone2_ckpt_path', type=str, default="")
    
    
    # #### MultiBranch Model
    # group.add_argument('--multi_branch_order' , type=str, default=None)
    # group.add_argument('--multibranch_select', type=str, default=None)


    # ### PatchModel ###<---- due to the dataclass limit it is impossible to parse a list from commend line, thus declear it here
    group.add_argument('--patch_range', type=str,default=None)

    # ### GraphCastModel
    # group.add_argument('--graphflag', type=str, default="mesh5")
    # group.add_argument('--agg_way', default='mean', type=str)

    # ### TimeSeries Model
    # group.add_argument('--canonical_fft', default=1, type=int)
    # group.add_argument('--pred_len', type=int, default=1)
    # group.add_argument('--label_len', type=int, default=3)
    return parser

def get_model_args(args):
    model_params = argparse.Namespace(
        img_size = args.img_size,
        history_length = args.history_length,
        in_chans = args.in_chans,
        out_chans = args.out_chans,
        embed_dim = args.embed_dim,
        depth     = args.depth,
        debug_mode = args.debug
    )

    return model_params

##############################################
############### Plugin Setting ################
##############################################
def _add_plugin_args(parser):
    group = parser.add_argument_group(title='plugin')
    group.add_argument('--GDMod_type', type=str, default='off')
    group.add_argument('--GDMod_lambda1', type=float, default=1)
    group.add_argument('--GDMod_lambda2', type=float, default=0)
    group.add_argument('--GDMod_L1_level', type=float, default=1)
    group.add_argument('--GDMod_L2_level', type=float, default=1)
    group.add_argument('--GDMod_sample_times', type=int, default=100)
    group.add_argument('--path_length_mode', type=str, default="000")
    group.add_argument('--rotation_regular_mode', type=str, default=None)
    group.add_argument('--rotation_regularize',action='store_true', default=False)
    group.add_argument('--GDMod_intervel', type=int, default=10)
    group.add_argument('--ngmod_freq', type=int,default=10, help='ngmod_freq')
    group.add_argument('--split_batch_chunk', type=int,default=16, help='split_batch_chunk')
    group.add_argument('--gmod_update_mode', type=int,default=2, help='gmod_update_mode')
    group.add_argument('--gd_alpha', type=float, default=1)
    group.add_argument('--gd_loss_wall', type=float, default=0)
    group.add_argument('--gd_loss_target', type=float, default=None)

    group.add_argument('--path_length_regularize', type=int, default=0)
    group.add_argument('--gmod_coef', type=str, default=None)
    group.add_argument('--gdamp', type=int, default=0)
    group.add_argument('--gdeval', type=int, default=0)

    group.add_argument('--consistancy_alpha', type=str, default=None)
    group.add_argument('--vertical_constrain', type=float, default=None)
    group.add_argument('--consistancy_activate_wall', default=100, type=float)
    group.add_argument('--consistancy_eval', default=0, type=int)
    group.add_argument('--consistancy_cut_grad', action='store_true', default=False)
    group.add_argument('--use_scalar_advection', action='store_true', default=False)
    group.add_argument('--gd_alpha_stratagy', type=str, default='normal')
    return parser    
def get_plugin_args(args):
    model_params = argparse.Namespace(
        img_size = args.img_size,
        patch_size = args.patch_size,
        history_length = args.history_length,
        in_chans = args.in_chans,
        out_chans = args.out_chans,
        embed_dim = args.embed_dim,
        depth = args.depth,
        debug_mode = args.debug_mode,
    )

    return model_params


##############################################
######## Data Augmentation Setting #########
##############################################
def _add_augmentation_args(parser):
    group = parser.add_argument_group(title='augmentation')
    group.add_argument('--input_noise_std', type=float,default=0.0, help='input_noise_std')
    return parser


##############################################
############## Train Setting #################
##############################################

def _add_training_args(parser):
    group = parser.add_argument_group(title='train')
    group.add_argument('--debug'                 , type=int, default=0, help='debug mode')
    group.add_argument('--mode'                  , type=str, default='pretrain', choices=['pretrain', 'finetune', 'more_epoch_train', 'continue_train'])
    group.add_argument('--batch_size'            , default=32, type=int)
    group.add_argument('--epochs'                , default=100, type=int)
    group.add_argument('--seed'                  , default=1994, type=int)
    group.add_argument('--accumulation_steps'    , type=int,default=1, help='accumulation_steps')
    group.add_argument('--skip_first_valid'      , action='store_true',default=False)
    group.add_argument('--do_first_fourcast'     , action='store_true', default=False)
    group.add_argument('--do_final_fourcast'     , action='store_true',default=False)
    group.add_argument('--do_fourcast_anyway'    , action='store_true',default=False)
    group.add_argument('--train_not_shuffle'     , action='store_true',default=False)
    group.add_argument('--load_model_lossy'      , action='store_true', default=False)
    group.add_argument('--find_unused_parameters', action='store_true', default=False)
    return parser

def get_train_args(args):
    training_params=argparse.Namespace(
        debug  = args.debug,
        mode  = args.Train.mode,
        batch_size  = args.batch_size,
        valid_batch_size  = args.valid_batch_size,
        epochs  = args.Train.epochs,
        seed  = args.Train.seed,
        accumulation_steps  = args.accumulation_steps,
        skip_first_valid  = args.skip_first_valid,
        do_first_fourcast  = args.do_first_fourcast,
        do_final_fourcast  = args.do_final_fourcast,
        do_fourcast_anyway  = args.do_fourcast_anyway,
        train_not_shuffle  = args.Train_not_shuffle,
        load_model_lossy  = args.load_model_lossy,
        find_unused_parameters  = args.find_unused_parameters,
    )
    return training_params



##############################################
############## ForeCast Setting ##############
##############################################
def _add_valid_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--forecast_every_epoch', type=int, default=0,help='forecast_every_epoch')
    group.add_argument('--pretrain_weight', type=str,default='', help='pretrain_weight')
    group.add_argument('--fourcast_randn_repeat', default=0, type=int)
    group.add_argument('--force_fourcast', default=0, type=int)
    group.add_argument('--snap_index'    , type=str, default=None)
    group.add_argument('--wandb_id'      , type=str, default=None)
    return parser

def get_valid_args(args):
    validation_params=argparse.Namespace(
        valid_batch_size= args.valid_batch_size,
        valid_every_epoch=args.valid_every_epoch,
        forecast_every_epoch=args.forecast_every_epoch,
        evaluate_every_epoch=args.evaluate_every_epoch,
        sampling_rate=args.sampling_rate,
        evaluate_branch=args.evaluate_branch
    )
    return validation_params


##############################################
############## Valid Setting #################
##############################################
def _add_valid_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--valid_batch_size', type=int, default=2,help='valid batch size')
    group.add_argument('--valid_every_epoch', type=int, default=1,help='valid_every_epoch')
    group.add_argument('--forecast_every_epoch', type=int, default=0,help='forecast_every_epoch')
    group.add_argument('--evaluate_every_epoch', type=int,
                       default=10, help='evaluate_every_epoch')
    group.add_argument('--evaluate_branch', type=str,
                       default='TEST', help='evaluate_branch')
    group.add_argument('--sampling_rate', type=int, default=100,help='sampling_rate')
    return parser
def get_valid_args(args):
    validation_params=argparse.Namespace(
        valid_batch_size= args.valid_batch_size,
        valid_every_epoch=args.valid_every_epoch,
        forecast_every_epoch=args.forecast_every_epoch,
        evaluate_every_epoch=args.evaluate_every_epoch,
        sampling_rate=args.sampling_rate,
        evaluate_branch=args.evaluate_branch
    )
    return validation_params

##############################################
############### Loss Setting #################
##############################################
def _add_loss_args(parser):
    group = parser.add_argument_group(title='loss')
    parser.add_argument('--criterion', type=str,default='mse', help='criterion')
    return parser
def _structure_loss_args(args):
    criterion_params = argparse.Namespace(
        criterion= args.Loss.criterion,
    )
    return criterion_params


##############################################
############## Monitor Setting ###############
##############################################
def _add_monitor_args(parser):
    group = parser.add_argument_group(title='monitor')
    group.add_argument('--use_wandb', type=str, default="off",help='when to activate wandb')
    group.add_argument("--tracemodel", type=int, default=0)
    group.add_argument("--log_trace_times", type=int, default=40)
    group.add_argument('--do_iter_log',action='store_true', help='whether continue train')
    group.add_argument('--disable_progress_bar',action='store_true', help='whether continue train')
    group.add_argument('--do_error_propagration_monitor', action='store_true')
    return parser
def get_monitor_args(args):
    monitor_params = argparse.Namespace(
        recorder_list= args.recorder_list,
        disable_progress_bar=args.disable_progress_bar,
        log_trace_times=args.log_trace_times,
        do_iter_log=args.do_iter_log,
        tracemodel=args.tracemodel
    )
    return monitor_params
##############################################
############## Dataset Setting ###############
##############################################
def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')
    group.add_argument('--epoch_save_list', type=int, nargs=',', default=None)
    group.add_argument('--save_every_epoch', default=1, type=int)
    group.add_argument('--save_warm_up', default=5, type=int)
    return parser


##############################################
############# Optimizer Setting ##############
##############################################
def _add_optimizer_args(parser):
    group = parser.add_argument_group(title='learning rate')
    # Optimizer parameters # feed into timm
    group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--clip_grad', type=float, default=0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    group.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 5e-4)')
    return parser
def get_optim_args(args):
    optimizer_params = argparse.Namespace(
        epochs=args.Train.epochs,
        opt=args.Optimizer.opt,
        opt_eps=args.opt_eps,
        opt_betas=args.opt_betas,
        clip_grad=args.clip_grad,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr=args.Optimizer.lr,
    )
    return optimizer_params

##############################################
############# Optimizer Setting ##############
##############################################
def _add_scheduler_args(parser):
    group = parser.add_argument_group(title='scheduler')
    group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    group.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    group.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    group.add_argument('--scheduler_inital_epochs', type=int, default=None)
    group.add_argument('--scheduler_min_lr', type=float, default=None)
    return parser
def get_scheduler_args(args):
    optimizer_params = argparse.Namespace(
        lr=args.Optimizer.lr,
        sched = args.Scheduler.sched,
        lr_noise = args.lr_noise,
        lr_noise_pct = args.lr_noise_pct,
        lr_noise_std = args.lr_noise_std,
        warmup_lr = args.warmup_lr,
        min_lr = args.min_lr,
        decay_epochs = args.decay_epochs,
        warmup_epochs = args.warmup_epochs,
        cooldown_epochs = args.cooldown_epochs,
        patience_epochs = args.patience_epochs,
        decay_rate = args.decay_rate,
        scheduler_inital_epochs = args.scheduler_inital_epochs,
        scheduler_min_lr = args.scheduler_min_lr
    )
    return optimizer_params



##############################################
############## Dataset Setting ###############
##############################################
def _add_dataset_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument('--root', type=str, default = 'datasets/WeatherBench/weatherbench32x64_1hour/')
    group.add_argument('--time_unit' , type = int, default = 1)
    group.add_argument('--img_size', type=int, nargs=',', default=(32,64))
    group.add_argument('--channel_name_list' , type = str, default="configs/datasets/WeatherBench/2D70.channel_list.json")
    group.add_argument('--timestamps_list' , type = int , default = None)
    group.add_argument('--time_step' , type = int , default = 2)
    group.add_argument('--time_intervel' , type = int , default = 1)
    group.add_argument('--dataset_patch_range' , type = int , nargs=',' , default=None)
    group.add_argument('--normlized_flag' , type = str, default = 'N') 
    group.add_argument('--time_reverse_flag' , type = str, default ='only_forward')
    group.add_argument('--use_time_feature' , action='store_true', default = False) 
    group.add_argument('--add_LunaSolarDirectly' , action='store_true', default = False) 
    group.add_argument('--offline_data_is_already_normed' , action='store_true', default = False)
    group.add_argument('--cross_sample' , action='store_true', default = False) 
    group.add_argument('--constant_channel_pick' , type = int, nargs=',' , default = None)
    group.add_argument('--make_data_physical_reasonable_mode' , type = str, default = None)
    group.add_argument('--share_memory', action='store_true', default = False, help='share_memory_flag')
    group.add_argument('--random_dataset', action='store_true', default = False, help='activate randomlized dataset ')
    group.add_argument('--num_workers', type=int, default=0, help='num worker is better set 0')
    group.add_argument('--use_offline_data', action='store_true', default = False)
    group.add_argument('--chunk_size', type=int, default=1024)
    group.add_argument('--picked_inputoutput_property',type=str, default=None)
    group.add_argument('--random_time_step', action='store_true', default=None)
    return parser

def get_data_args(args):
    data_params = argparse.Namespace(
        root = args.root,
        time_unit = args.time_unit,
        resolution_w = args.img_size[1],
        resolution_h = args.img_size[0],
        channel_name_list = args.channel_name_list,
        timestamps_list = args.timestamps_list,
        time_step = args.Dataset.time_step,
        time_intervel = args.time_intervel,
        normlized_flag = args.normlized_flag,
        time_reverse_flag = args.time_reverse_flag,
        use_time_feature = args.use_time_feature,
        add_LunaSolarDirectly = args.add_LunaSolarDirectly,
        offline_data_is_already_normed = args.offline_data_is_already_normed,
        cross_sample = args.cross_sample,
        constant_channel_pick = args.constant_channel_pick,
        make_data_physical_reasonable_mode         = args.make_data_physical_reasonable_mode     
    )
    return data_params

if __name__ == '__main__':
    args = get_args()
    #
