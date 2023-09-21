
import argparse
from .base import Config
import time
def _print_args(args):
    """Print arguments."""
    if args.Pengine.engine.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


#--------------------------------------------------#
def print_namespace_tree(namespace, indent=0):
    namespace = vars(namespace) if isinstance(
        namespace, (argparse.Namespace, Config)) else namespace
    for key, value in namespace.items():
        print(' ' * indent, end='')
        if isinstance(value, (dict, (argparse.Namespace, Config))):
            print(key)
            print_namespace_tree(value, indent + 4)
        else:
            print(f"{key:30s} ---> {value}")

def convert_namespace_tree(namespace):
    namespace = vars(namespace) if isinstance(namespace, argparse.Namespace) else namespace
    if isinstance(namespace,dict):
        return dict([(key, convert_namespace_tree(val)) for key, val in namespace.items()])
    else:
        return namespace

def tuple2str(_tuple):
    if isinstance(_tuple,(list,tuple)):
        return '.'.join([str(t) for t in (_tuple)])
    else:
        return _tuple

def get_model_name(args):
    model_name = args.Model.model.get_name()
    # if "FED" in args.Model.model.model_type:
    #     mode_name =args.modes.replace(",","-")
    #     return f"{args.Model.model.model_type}.{args.mode_select}_select.M{mode_name}_P{args.pred_len}L{args.label_len}"
    # if "AFN" in args.Model.model.model_type and hasattr(args,'model_depth') and args.Model.model.model_depth == 6:
    #     model_name = "small_" + model_name
    # model_name = f"ViT_in_bulk-{model_name}" if len(args.img_size)>2 else model_name
    # model_name = f"{args.wrapper_model}-{model_name}" if args.wrapper_model else model_name
    # model_name = f"{model_name}_Patch_{tuple2str(args.patch_range)}" if (args.patch_range and 'Patch' in args.Dataset.dataset.dataset_type) else model_name
    return model_name

def get_datasetname(args):
    datasetname = "test"
    # if not args.Dataset.dataset.dataset_type and args.Train_set in train_set:
    #     datasetname = train_set[args.Train_set][4].__name__
    # if not datasetname:
    #     raise NotImplementedError("please use right dataset type")
        
    # if datasetname in ["",'ERA5CephDataset','ERA5CephSmallDataset']:
    #     datasetname  = "ERA5_20-12" if 'physics' in args.Train_set else "ERA5_20"
    return datasetname

def get_projectname(args):
    model_name   = get_model_name(args)
    datasetname  = get_datasetname(args)
    project_name = f"{args.Train.mode}-test"
    # if "Self" in datasetname:
    #     property_names = 'UVTPH'
    #     picked_input_name = "".join([property_names[t] for t in args.picked_input_property])
    #     picked_output_name= "".join([property_names[t] for t in args.picked_output_property])
    #     project_name = f"{picked_input_name}_{picked_output_name}"
    # else:
    #     project_name = f"{args.Train.mode}-{args.Train_set}"
    #     if hasattr(args,'random_time_step') and args.random_time_step:project_name = 'rd_sp_'+project_name 
    #     if hasattr(args,'time_step') and args.Dataset.time_step:              project_name = f"ts_{args.Dataset.time_step}_" +project_name 
    #     if hasattr(args,'history_length') and args.history_length !=1:project_name = f"his_{args.history_length}_"+project_name
    #     if hasattr(args,'time_reverse_flag') and args.time_reverse_flag !="only_forward":project_name = f"{args.time_reverse_flag}_"+project_name
    #     if hasattr(args,'time_intervel') and args.time_intervel:project_name = project_name + f"_per_{args.time_intervel}_step"
    #     if args.patch_range != args.dataset_patch_range and args.patch_range and args.dataset_patch_range: 
    #         project_name = project_name + f"_P{tuple2str(args.dataset_patch_range)}_for_P{tuple2str(args.patch_range)}"
    #     #if hasattr(args,'cross_sample') and args.cross_sample:project_name = project_name + f"_random_dataset"
    #     #print(project_name)
    return model_name, datasetname,project_name

def deal_with_tuple_string(patch_size,defult=None,dtype=int):
    if isinstance(patch_size,str):
        if len(patch_size)>0:
            patch_size  = tuple([dtype(t) for t in patch_size.split(',')])
            if len(patch_size) == 1: patch_size=patch_size[0]
        else:
            patch_size = defult
    elif isinstance(patch_size,int):
        patch_size = patch_size
    elif isinstance(patch_size,list):
        patch_size = tuple(patch_size)
    elif isinstance(patch_size,tuple):
        pass
    else:
        patch_size = defult
    return patch_size

import random, os
def get_ckpt_path(args):
    if args.debug:return './debug'
    #TIME_NOW  = time.strftime("%m_%d_%H_%M")+f"_{args.port}" if args.Pengine.engine.distributed else time.strftime("%m_%d_%H_%M_%S")
    TIME_NOW = time.strftime("%m_%d_%H_%M")
    if args.Train.seed == -1:args.Train.seed = 42;#random.randint(1, 100000)
    if args.Train.seed == -2:args.Train.seed = random.randint(1, 100000)
    TIME_NOW  = f"{TIME_NOW}-seed_{args.Train.seed}"
    args.trail_name = TIME_NOW
    if not hasattr(args,'train_set'):args.Train_set='large'
    model_name, datasetname, project_name = get_projectname(args)
    if args.Train.mode in ['continue_train', 'fourcast'] and (not args.Train.do_fourcast_anyway):
        assert args.Checkpoint.pretrain_weight
        #args.Train.mode = "finetune"
        SAVE_PATH = os.path.dirname(args.Checkpoint.pretrain_weight)
    else:
        SAVE_PATH = f'./checkpoints/{datasetname}/{model_name}/{project_name}/{TIME_NOW}'
        os.makedirs(SAVE_PATH, exist_ok=True)
    return SAVE_PATH

def parse_default_args(args):
    if args.Train.seed == -1:args.Train.seed = 42
    if args.Train.seed == -2:args.Train.seed = random.randint(1, 100000)
    args.half_model = half_model
    args.Train.batch_size = bs_for_mode[args.Train.mode] if args.Train.batch_size == -1 else args.Train.batch_size
    args.Valid.valid_batch_size = args.Train.batch_size*8 if args.Valid.valid_batch_size == -1 else args.Valid.valid_batch_size
    args.Train.epochs     = ep_for_mode[args.Train.mode] if args.Train.epochs == -1 else args.Train.epochs
    args.Optimizer.lr         = lr_for_mode[args.Train.mode] if args.Optimizer.lr == -1 else args.Optimizer.lr
    args.Dataset.time_step = ts_for_mode[args.Train.mode] if args.Dataset.time_step == -1 else args.Dataset.time_step

    if not hasattr(args,'epoch_save_list'):args.Checkpoint.epoch_save_list = [99]
    if isinstance(args.Checkpoint.epoch_save_list,str):args.Checkpoint.epoch_save_list = [int(p) for p in args.Checkpoint.epoch_save_list.split(',')]
    # input size
    img_size = patch_size = x_c = y_c =  dataset_type = None

    if args.Train_set is not None and args.Train_set in train_set:
        img_size, patch_size, x_c, y_c, dataset_type,dataset_kargs = train_set[args.Train_set]
        
        if 'Euler' in args.wrapper_model: y_c  = 15
    else:
        assert args.img_size
        assert args.patch_size
        assert args.input_channel
        assert args.output_channel
        assert args.Dataset.dataset.dataset_type
        dataset_kargs={}


    dataset_kargs['root'] = args.data_root if args.data_root != "" else None
    dataset_kargs['mode']        = args.Train.mode
    dataset_kargs['time_step']   = args.Dataset.time_step
    dataset_kargs['check_data']  = True
    dataset_kargs['time_reverse_flag'] = 'only_forward' if not hasattr(args,'time_reverse_flag') else args.time_reverse_flag
    
    dataset_kargs['use_offline_data'] = args.use_offline_data
    dataset_kargs['add_ConstDirectly'] = args.add_ConstDirectly
    dataset_kargs['add_LunaSolarDirectly'] = args.add_LunaSolarDirectly
    if hasattr(args,'dataset_flag') and args.dataset_flag:dataset_kargs['dataset_flag']= args.dataset_flag
    if hasattr(args,'time_intervel'):dataset_kargs['time_intervel']= args.time_intervel
    if hasattr(args,'cross_sample'):dataset_kargs['cross_sample']= args.cross_sample
    if hasattr(args,'use_time_stamp') and args.use_time_stamp:dataset_kargs['use_time_stamp']= args.use_time_stamp
    if hasattr(args,'use_position_idx'):dataset_kargs['use_position_idx']= args.use_position_idx
    
    args.unique_up_sample_channel = args.unique_up_sample_channel
    

    args.Dataset.dataset.dataset_type = dataset_type if not args.Dataset.dataset.dataset_type else args.Dataset.dataset.dataset_type
    args.Dataset.dataset.dataset_type = args.Dataset.dataset.dataset_type.__name__ if not isinstance(args.Dataset.dataset.dataset_type,str) else args.Dataset.dataset.dataset_type
    x_c        = args.input_channel = x_c if not args.input_channel else args.input_channel
    y_c        = args.output_channel= y_c if not args.output_channel else args.output_channel
    patch_size = args.patch_size = deal_with_tuple_string(args.patch_size,patch_size)
    img_size   = args.img_size   = deal_with_tuple_string(args.img_size,img_size)
    patch_range= args.patch_range= deal_with_tuple_string(args.patch_range,None)
    multibranch_select = args.multibranch_select= deal_with_tuple_string(args.multibranch_select,None)
    #print(args.multibranch_select)
    if "Patch" in args.Dataset.dataset.dataset_type:
        dataset_patch_range = args.dataset_patch_range = deal_with_tuple_string(args.dataset_patch_range,None)
    else:
        dataset_patch_range = args.dataset_patch_range = None
    dataset_kargs['img_size'] = img_size
    dataset_kargs['patch_range']= dataset_patch_range if dataset_patch_range else patch_range
    dataset_kargs['debug']= args.debug
    dataset_kargs['multibranch_select']= args.multibranch_select
    args.dataset_kargs = dataset_kargs
    args.picked_input_property = args.picked_output_property = None
    if args.picked_inputoutput_property:
        args.picked_input_property, args.picked_output_property = args.picked_inputoutput_property.split(".")
        args.picked_input_property = deal_with_tuple_string(args.picked_input_property,None)
        args.picked_input_property = [args.picked_input_property] if isinstance(args.picked_input_property,int) else args.picked_input_property
        args.picked_output_property = deal_with_tuple_string(args.picked_output_property,None)
        args.picked_output_property= [args.picked_output_property] if isinstance(args.picked_output_property,int) else args.picked_output_property
        args.input_channel = 14*len(args.picked_input_property)
        args.output_channel= 14*len(args.picked_output_property)
    dataset_kargs['picked_input_property'] = args.picked_input_property
    dataset_kargs['picked_output_property'] = args.picked_output_property
    # model_img_size= args.img_size
    # if 'Patch' in args.wrapper_model:
    #     if '3D' in args.wrapper_model:
    #         model_img_size = tuple([5]*3)
    #     else:
    #         model_img_size = tuple([5]*2)
    model_kargs={
        "img_size": args.img_size, 
        "patch_size": args.patch_size, 
        "patch_range":args.patch_range,
        "in_chans": args.input_channel, 
        "out_chans": args.output_channel,
        "fno_blocks": args.fno_blocks,
        "embed_dim": args.embed_dim if not args.debug else 16*6, 
        "depth": args.Model.model.model_depth if not args.debug else 1,
        "debug_mode":args.debug,
        "double_skip":args.double_skip, 
        "fno_bias": args.fno_bias, 
        "fno_softshrink": args.fno_softshrink,
        "history_length": args.history_length,
        "reduce_Field_coef":args.use_scalar_advection,
        "modes":deal_with_tuple_string(args.modes,(17,33,6)),
        "mode_select":args.mode_select,
        "physics_num":args.physics_num,
        "pred_len":args.pred_len,
        "n_heads":args.n_heads,
        "label_len":args.label_len,
        "canonical_fft":args.canonical_fft,
        "unique_up_sample_channel":args.unique_up_sample_channel,
        "share_memory":args.share_memory,
        "dropout_rate":args.dropout_rate,
        "conv_simple":args.conv_simple,
        "graphflag":args.graphflag,
        "use_pos_embed":args.use_pos_embed,
        "agg_way":args.agg_way
    }
    args.Model.model.model_kargs = model_kargs


    args.snap_index = [[0,40,80,12], [t for t in [38,49,13,27] if t < args.output_channel]      # property  Z500 and T850 and v2m and u2m and 
                       ] 
    if args.wrapper_model == 'PatchWrapper':
        args.snap_index.append({0:[[15],[15]],1:[[13],[15]],2:[[11],[15]],3:[[ 9],[15]],4:[[ 7],[15]],5:[[ 5],[15]]})
    else:
        args.snap_index.append([[15,15,15, 7, 7, 7,23,23,23],
                                [15,31,45,15,31,45,15,31,45]])
    if args.output_channel<=13:args.snap_index=None
    if not hasattr(args,'ngpus_per_node'):args.Pengine.engine.ngpus_per_node=1
    args.real_batch_size = args.Train.batch_size * args.accumulation_steps * args.Pengine.engine.ngpus_per_node 
    args.compute_graph = parser_compute_graph(args.compute_graph_set)
    args.torch_compile = (torch.__version__[0]=="2" and args.torch_compile)
    return args

def parser_compute_graph(compute_graph_set):
    # X0 X1 X2
    # |  |  |
    # x1 x2 x3
    # |  |
    # y2 y3
    # |
    # z3

    if compute_graph_set is None:
        return None, None
    if compute_graph_set == "":
        return None, None
    compute_graph_set_pool = {
        'fwd3_D': ([[1], [2], [3]], [[0, 1, 1, 0.33, "quantity"],
                                     [0, 2, 2, 0.33, "quantity"],
                                     [0, 3, 3, 0.33, "quantity"]]),
        'fwd3_D_Rog5': ([[1], [2], [3]], [[0, 1, 1, 1.0, "quantity_real_log5"], [0, 2, 2, 1.0, "quantity_real_log5"], [0, 3, 3, 1.0, "quantity_real_log5"]]),
        'fwd3_D_Mog': ([[1], [2], [3]], [[0, 1, 1, 1.0, "quantity_mean_log"], [0, 2, 2, 1.0, "quantity_mean_log"], [0, 3, 3, 1.0, "quantity_mean_log"]]),
        'fwd2_TA': ([[1, 2, 3], [2], [3]], [[0, 1, 1, 0.25, "quantity"],
                                            [0, 2, 2, 0.25, "quantity"],
                                            [1, 2, 2, 0.25, "alpha"],
                                            [1, 3, 3, 0.25, "alpha"]
                                            ]),
        'fwd2_TAL': ([[1, 2, 3], [2], [3]], [[0, 1, 1, 0.25, "quantity"],
                                             [0, 2, 2, 0.25, "quantity"],
                                             [1, 2, 2, 0.25, "alpha_log"],
                                             [1, 3, 3, 0.25, "alpha_log"]
                                             ]),
        'fwd2_KAR': ([[1, 2, 3], [2, 3], [3]], [[0, 1, 1, 0.5, "quantity"],
                                                [0, 2, 2, 0.5, "quantity"],
                                                [1, 2, 2, 0.5, "quantity"],
                                                [1, 3, 3, 0.5, "quantity"],
                                                [2, 3, 3, 0.5, "quantity"]
                                                ]),
        'fwd1_D': ([[1]],   [[0, 1, 1, 1.0, "quantity"]]),
        'fwd1_TA': ([[1, 2], [2]],   [[0, 1, 1, 1.0, "quantity"], [1, 2, 2, 1.0, "alpha"]]),
        'fwd2_D': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity"], [0, 2, 2, 1.0, "quantity"]]),
        'fwd2_D_Log': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_log"], [0, 2, 2, 1.0, "quantity_log"]]),
        'fwd2_D_Rog': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log"], [0, 2, 2, 1.0, "quantity_real_log"]]),
        'fwd2_D_Rog5': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log5"], [0, 2, 2, 1.0, "quantity_real_log5"]]),

        'fwd2_D_Rog3': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log3"], [0, 2, 2, 1.0, "quantity_real_log3"]]),
        'fwd2_D_Rog2': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log2"], [0, 2, 2, 1.0, "quantity_real_log2"]]),
        'fwd2_D_Mog': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_mean_log"], [0, 2, 2, 1.0, "quantity_mean_log"]]),
        'fwd2_D_Pog': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_batch_mean_log"], [0, 2, 2, 1.0, "quantity_batch_mean_log"]]),
        'fwd1_D_Mog': ([[1]],   [[0, 1, 1, 1.0, "quantity_mean_log"]]),
        'fwd1_D_Rog5': ([[1]],   [[0, 1, 1, 1.0, "quantity_real_log5"]]),
        'fwd2_P': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                   [0, 2, 2, 1.0, "quantity"],
                                   [1, 2, 2, 1.0, "quantity"]
                                   ]),
        'fwd2_PR': ([[1, 2], [2]], [[0, 1, 1, 0.5, "quantity"],
                                    [0, 2, 2, 0.5, "quantity"],
                                    [1, 2, 2, 1.0, "quantity"]
                                    ]),
        'fwd2_PRO': ([[1, 2], [2]], [[0, 1, 1, 1, "quantity"],
                                     [0, 2, 2, 1, "quantity"],
                                     [1, 2, 2, 0.5, "quantity"]
                                     ]),
        'fwd4_AC': ([[1, 2, 3, 4],
                     [2],
                     [3],
                     [4]],
                    [[0, 1, 1, 1, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                     [1, 4, 4, 1, "quantity"],
                     ]),
        'fwd4_KC_L': ([[1, 2, 3, 4],
                       [2],
                       [3],
                       [4]],
                      [[0, 3, 3, 1, "quantity"],
                          [1, 2, 2, 0.33, "quantity"],
                          [1, 3, 3, 0.33, "quantity"],
                          [1, 4, 4, 0.33, "quantity"],
                       ]),
        'fwd4_AC': ([[1, 2, 3, 4],
                     [2],
                     [3],
                     [4]],
                    [[0, 1, 1, 1, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                     [1, 4, 4, 1, "quantity"],
                     ]),
        'fwd4_C': ([[1, 2, 3, 4],
                    [2],
                    [3],
                    [4]],
                   [[0, 1, 1, 1, "quantity"],
                       [1, 4, 4, 1, "quantity"],
                    ]),
        'fwd4_ABC': ([[1, 2, 3, 4],
                      [2],
                      [3],
                      [4]],
                     [[0, 1, 1, 1, "quantity"],
                         [0, 1, 2, 1, "quantity"],
                         [0, 1, 3, 1, "quantity"],
                         [1, 2, 2, 1, "quantity"],
                         [1, 3, 3, 1, "quantity"],
                         [1, 4, 4, 1, "quantity"],
                      ]),
        'fwd4_ABC_H': ([[1, 2, 3, 4],
                        [2],
                        [3],
                        [4]],
                       [[0, 1, 1, 1, "quantity"],
                           [0, 1, 2, 1, "quantity"],
                           [0, 1, 3, 1, "quantity"],
                        [1, 2, 2, 2, "quantity"],
                        [1, 3, 3, 2, "quantity"],
                        [1, 4, 4, 2, "quantity"],
                        ]),
        'fwd4_ABC_L': ([[1, 2, 3, 4],
                        [2],
                        [3],
                        [4]],
                       [[0, 1, 1, 0.5, "quantity"],
                           [0, 1, 2, 0.5, "quantity"],
                           [0, 1, 3, 0.5, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                        [1, 4, 4, 1, "quantity"],
                        ]),
        'fwd3_ABC': ([[1, 2, 3],
                      [2],
                      [3]],
                     [[0, 1, 1, 1, "quantity"],
                      [0, 1, 2, 1, "quantity"],
                      [1, 2, 2, 1, "quantity"],
                      [1, 3, 3, 1, "quantity"]
                      ]),
        'fwd3_ABC_Log': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity_log"],
                          [0, 1, 2, 1, "quantity_log"],
                          [1, 2, 2, 1, "quantity_log"],
                          [1, 3, 3, 1, "quantity_log"]
                          ]),
        'fwd3_DC_Log': ([[1, 3],
                         [2],
                         [3]],
                        [[0, 1, 1, 1, "quantity_log"],
                         [0, 2, 2, 1, "quantity_log"],
                         [1, 3, 3, 1, "quantity_log"]
                         ]),
        'fwd3_D_Log': ([[1], [2], [3]],   [[0, 1, 1, 1.0, "quantity_log"], [0, 2, 2, 1.0, "quantity_log"], [0, 3, 3, 1.0, "quantity_log"]]),
        'fwd3_D_Pog': ([[1], [2], [3]],   [[0, 1, 1, 1.0, "quantity_batch_mean_log"], [0, 2, 2, 1.0, "quantity_batch_mean_log"], [0, 3, 3, 1.0, "quantity_batch_mean_log"]]),
        'fwd2_PA': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                    [0, 2, 2, 1.0, "quantity"],
                                    [1, 2, 2, 1.0, "alpha"]
                                    ]),
        'fwd2_PAL': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                     [0, 2, 2, 1.0, "quantity"],
                                     [1, 2, 2, 1.0, "alpha_log"]
                                     ]),
        'fwd3_DlongT5': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity"],
                          [0, 1, 2, 1, "quantity"],
                          [0, 1, 3, 1, "quantity"],
                          ], 5),  # <--- in old better version it is another mean
        'fwd3_longT10': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity"],
                         [0, 1, 2, 1, "quantity"],
                         [0, 1, 3, 1, "quantity"],
                         [0, 2, 2, 1, "quantity"],
                         [0, 3, 3, 1, "quantity"],
                          ], "during_valid_normal"),
        'fwd3_D_go10': ([[1], [2], [3]],
                        [],  # <--- no need, will auto deploy for during_valid_normal mode
                        "during_valid_normal_10"),
        'fwd3_D_go10_deltalog': ([[1], [2], [3]],
                                 [],  # <--- no need, will auto deploy for during_valid_normal mode
                                 "during_valid_deltalog_10"),
        'fwd3_D_go10_per_feature': ([[1], [2], [3]],
                                    [],  # <--- no need, will auto deploy for during_valid_normal mode
                                    "during_valid_per_feature_10"),
        'fwd3_D_go10_per_feature_needbase': ([[1], [2], [3]],
                                             [],  # <--- no need, will auto deploy for during_valid_normal mode
                                             "needbase_during_valid_per_feature_10"),
        'fwd3_D_go10_needbase': ([[1], [2], [3]],
                                 [],  # <--- no need, will auto deploy for during_valid_normal mode
                                 "needbase_during_valid_normal_10"),
        'fwd3_D_go10_vallina': ([[1], [2], [3]],
                                [],  # <--- no need, will auto deploy for during_valid_normal mode
                                "vallina_during_valid_normal_10"),
        'fwd3_D_go10_per_feature_vallina': ([[1], [2], [3]],
                                            [],  # <--- no need, will auto deploy for during_valid_normal mode
                                            "vallina_during_valid_per_feature_10"),
        'fwd3_D_go10_per_sample_vallina': ([[1], [2], [3]],
                                           [],  # <--- no need, will auto deploy for during_valid_normal mode
                                           "vallina_during_valid_per_sample_10"),
        'fwd3_D_go10_per_sample_logoffset': ([[1], [2], [3]],
                                             [],  # <--- no need, will auto deploy for during_valid_normal mode
                                             "logoffset_during_valid_per_sample_10"),
        'fwd3_D_go10_runtime_logoffset': ([[1], [2], [3]],
                                          [],  # <--- no need, will auto deploy for during_valid_normal mode
                                          "logoffset_runtime_10"),

    }

    return compute_graph_set_pool[compute_graph_set]
