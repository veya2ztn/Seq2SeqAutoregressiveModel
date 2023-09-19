import argparse
import json
from dataclasses import dataclass
from simple_parsing import ArgumentParser, subgroups, field
from model.model_arguements import (
    ModelConfig, AFNONetConfig, GraphCastConfig, PatchEmbeddingConfig)
from typing import Optional, List, Tuple, Union
from .base import Config
import yaml

def flatten_args(namespace, level=0):
    namespace = vars(namespace) if isinstance(
        namespace, (argparse.Namespace, Config)) else namespace
    if not isinstance(namespace, dict):
        return namespace

    out = {}
    for key, value in namespace.items():
        if key in ['config_path']:
            continue
        if level == 0 and isinstance(value, Config) and isinstance(list(vars(value).values())[0], Config):
            for k, v in vars(value).items():
                out[f"{key}.{k}"] = flatten_args(v, level+2)
        else:
            out[key] = flatten_args(value, level+1)

    return out


def save_args(args, path):
    with open(path, 'w') as f:
        yaml.dump(flatten_args(args), f, default_flow_style=False)

def build_parser():
    parser = ArgumentParser(description='Arguments', allow_abbrev=False, add_help=True,add_config_path_arg=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--distributed', action='store_true', default=False)
    
    parser.add_arguments(Global_Model_Config      ,dest = "Model")
    parser.add_arguments(Global_Train_Config      ,dest = "Train")
    parser.add_arguments(Global_Loss_Config       ,dest = "Loss")
    parser.add_arguments(Global_Valid_Config      ,dest = "Valid")
    parser.add_arguments(Global_Monitor_Config    ,dest = "Monitor")
    parser.add_arguments(Global_Checkpoint_Config ,dest = "Checkpoint")
    parser.add_arguments(Global_Optimizer_Config  ,dest = "Optimizer")
    parser.add_arguments(Global_Scheduler_Config  ,dest = "Scheduler")
    parser.add_arguments(Global_Dataset_Config    ,dest = "Dataset")
    parser.add_arguments(Global_Forecast_Config   ,dest = "Forecast")
    
    return parser

def get_plain_parser():
    parser = build_parser()
    return parser.parse_args()

def get_args_parser():
    """Parse all arguments in structure way"""
    args = get_plain_parser()
    return args


def get_args(config_path=None):
    if config_path is not None:
        import sys as _sys
        args = ['--config_path', 'test.yaml']+_sys.argv[1:]
    else:
        args = None
    parser = build_parser()
    return parser.parse_args(args=args)


@dataclass
class Global_Model_Config(Config):

    # Which model to use
    model: ModelConfig = subgroups(
        {"afnonet": AFNONetConfig, "graphcast": GraphCastConfig}
    )

    patch_embedding: PatchEmbeddingConfig = subgroups(
        {"simple": PatchEmbeddingConfig},
        default = 'simple'
    )


@dataclass
class Plugin_Config(Config):
    GDMod_type: str = field(default='off')
    GDMod_lambda1: float = field(default=1)
    GDMod_lambda2: float = field(default=0)
    GDMod_L1_level: float = field(default=1)
    GDMod_L2_level: float = field(default=1)
    GDMod_sample_times: int = field(default=100)
    path_length_mode: str = field(default="000")
    rotation_regular_mode: str = field(default=None)
    rotation_regularize: bool = field(default=False)
    GDMod_intervel: int = field(default=10)
    ngmod_freq: int = field(default=10, help='ngmod_freq')
    split_batch_chunk: int = field(default=16, help='split_batch_chunk')
    gmod_update_mode: int = field(default=2, help='gmod_update_mode')
    gd_alpha: float = field(default=1)
    gd_loss_wall: float = field(default=0)
    gd_loss_target: float = field(default=None)
    path_length_regularize: int = field(default=0)
    gmod_coef: str = field(default=None)
    gdamp: int = field(default=0)
    gdeval: int = field(default=0)
    consistancy_alpha: str = field(default=None)
    vertical_constrain: float = field(default=None)
    consistancy_activate_wall:int = field(default=100)
    consistancy_eval:bool = field(default=0)
    consistancy_cut_grad: bool = field(default=False)
    use_scalar_advection: bool = field(default=False)
    gd_alpha_stratagy: str = field(default='normal')

@dataclass
class Global_Train_Config(Config):

    batch_size            :int = field(default=32   )
    epochs                :int = field(default=100  )
    seed                  :int = field(default=1994 )
    accumulation_steps    :int = field(default=1)
    mode                  :str = field(default='pretrain', choices=['pretrain', 'finetune', 'more_epoch_train', 'continue_train'])
    skip_first_valid      :bool = field(default=False) 
    do_first_fourcast     :bool = field(default=False) 
    do_final_fourcast     :bool = field(default=False) 
    do_fourcast_anyway    :bool = field(default=False) 
    train_not_shuffle     :bool = field(default=False) 
    load_model_lossy      :bool = field(default=False) 
    find_unused_parameters:bool = field(default=False) 
    input_noise_std: float = field(default=0.0)
    compute_graph_set: str = field(default=None)
@dataclass
class Global_Forecast_Config(Config):
    forecast_every_epoch  :int=field(default=0,help='forecast_every_epoch')
    fourcast_randn_repeat :int=field(default=False, help='add random noise when do forecast, now disable')
    force_fourcast        :bool=field(default=False) 
    pretrain_weight       :Optional[str]=field(default=None, help='pretrain_weight')
    snap_index            :Optional[str]=field(default=None)
    wandb_id              :Optional[str]=field(default=None)

@dataclass
class Global_Valid_Config(Config):
    valid_batch_size     :int=field(default=2,help='valid batch size')
    valid_every_epoch    :int=field(default=1,help='valid_every_epoch')
    evaluate_every_epoch :int=field(default=10, help='evaluate_every_epoch')
    evaluate_branch      :str=field(default='TEST', help='evaluate_branch')


@dataclass
class Global_Loss_Config(Config):
    criterion: str = field(default='mse')
    

@dataclass
class Global_Monitor_Config(Config):
    use_wandb       :str=field(default="off", help='when to activate wandb')
    tracemodel      :int=field(default=0)
    log_trace_times :int=field(default=40)
    recorder_list   :List[str]=field(default_factory=lambda: ['tensorboard'])
    do_iter_log                   :bool=field(default=False)
    disable_progress_bar          :bool=field(default=False)
    do_error_propagration_monitor :bool=field(default=False)


@dataclass
class Global_Checkpoint_Config(Config):
    epoch_save_list   :Optional[List[int]] = field(default=None)   
    save_every_epoch  :int = field(default = 1)    
    save_warm_up      :int = field(default = 5)

@dataclass
class Global_Optimizer_Config(Config):
    opt:str=field(default='adamw', help='Optimizer (default: "adamw"')
    opt_eps:float=field(default=1e-8, help='Optimizer Epsilon (default: 1e-8)')
    opt_betas:float=field(default=None, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    clip_grad:float=field(default=0, help='Clip gradient norm (default: None, no clipping)')
    momentum:float=field(default=0.9, help='SGD momentum (default: 0.9)')
    weight_decay:float=field(default=0.05, help='weight decay (default: 0.05)')
    lr:float=field(default=1e-4, help='learning rate (default: 5e-4)')


@dataclass
class Global_Scheduler_Config(Config):
    sched :str=field(default='cosine', help='LR scheduler (default: "cosine"')
    lr_noise :Optional[float] = field(default=None, help='learning rate noise on/off epoch percentages')
    lr_noise_pct :float=field(default=0.67, help='learning rate noise limit percent (default: 0.67)')
    lr_noise_std :float=field(default=1.0, help='learning rate noise std_dev (default: 1.0)')
    warmup_lr :float=field(default=1e-6, help='warmup learning rate (default: 1e_6)')
    min_lr :float=field(default=1e-5, help='lower lr bound for cyclic schedulers that hit 0 (1e_5)')
    decay_epochs :float=field(default=30, help='epoch interval to decay LR')
    warmup_epochs :int=field(default=5, help='epochs to warmup LR, if scheduler supports')
    cooldown_epochs :int=field(default=10, help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    patience_epochs :int=field(default=10, help='patience epochs for Plateau LR scheduler (default: 10')
    decay_rate :float=field(default=0.1, help='LR decay rate (default: 0.1)')
    scheduler_inital_epochs :int=field(default=None)
    scheduler_min_lr :float=field(default=None)

@dataclass
class Global_Dataset_Config(Config):
    root:str=field(default='datasets/WeatherBench/weatherbench32x64_1hour/')
    time_unit:int=field(default=1)
    dataset_patch_range:Optional[List[int]]=field(default=None)
    constant_channel_pick: Optional[List[int]] = field(default=None)
    channel_name_list:str=field(default="configs/datasets/WeatherBench/2D70.channel_list.json")
    timestamps_list:int=field(default=None)
    time_step:int=field(default=2)
    time_intervel:int=field(default=1)
    normlized_flag:str=field(default='N')
    time_reverse_flag:str=field(default='only_forward')
    use_time_feature:bool=field(default=False)
    add_LunaSolarDirectly:bool=field(default=False)
    offline_data_is_already_normed:bool=field(default=False)
    cross_sample:bool=field(default=False)
    make_data_physical_reasonable_mode:str=field(default=None)
    share_memory:bool=field(default=False,help='share_memory_flag')
    random_dataset:bool=field(default=False,help='activaterandomlizeddataset')
    num_workers:int=field(default=0,help='numworkerisbetterset0')
    use_offline_data:bool=field(default=False)
    chunk_size:int=field(default=1024)
    picked_inputoutput_property:str=field(default=None)
    random_time_step:bool=field(default=None)


@dataclass
class Global_Config(Config):

    train: Global_Train_Config
    criterion: Global_Loss_Config
    valid: Global_Valid_Config
    monitor: Global_Monitor_Config
    checkpoint: Global_Checkpoint_Config
    optimizer: Global_Optimizer_Config
    scheduler: Global_Scheduler_Config
    dataset: Global_Dataset_Config
    forecast: Global_Forecast_Config
    model:  ModelConfig = subgroups(
        {"afnonet": AFNONetConfig, "graphcast": GraphCastConfig}, default="afnonet"
    )

if __name__ == '__main__':
    args = get_args()
    #
