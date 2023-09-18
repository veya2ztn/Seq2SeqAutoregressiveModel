

import torch
import torch.backends.cudnn as cudnn
#from utils import load_model, getModelSize, deal_with_tuple_string
import timm
import os
import numpy as np


def build_combination_model(args):
    args.model_kargs['history_length'] = 1
    assert args.model_type1
    assert args.model_type2
    args.model_kargs['in_chans']   = eval(args.wrapper_model).default_input_channel1
    args.model_kargs['out_chans']  = eval(args.wrapper_model).default_output_channel1
    backbone1                      = eval(args.model_type1)(**args.model_kargs)
        
    
    args.model_kargs['in_chans']  = eval(args.wrapper_model).default_input_channel2
    args.model_kargs['out_chans'] = eval(args.wrapper_model).default_output_channel2

    if args.model_type2 == 'AFNONet':
        pass
    elif args.model_type2 == 'smallAFNONet':
        args.model_kargs['depth'] = 6
        args.model_type2 = 'AFNONet'
    elif args.model_type2 == 'tinyAFNONet':
        args.model_kargs['embed_dim'] = 384
        args.model_kargs['depth'] = 6
        args.model_type2 = 'AFNONet'
    else:
        raise NotImplementedError

    backbone2 = eval(args.model_type2)(**args.model_kargs)
    args.model_kargs['in_chans'] = args.input_channel
    args.model_kargs['out_chans'] = args.output_channel
    args.model_kargs['history_length'] = 1
    model = eval(args.wrapper_model)(args, backbone1, backbone2, args.backbone1_ckpt_path, args.backbone2_ckpt_path)
    return model
    
def build_wrapper_model(args):
    model = eval(args.model_type)(**args.model_kargs)
    if args.wrapper_model:
        if args.subweight:
            print(f"in wrapper model, load subweight from {args.subweight}")
            load_model(model, path=args.subweight, only_model=True,loc='cpu', strict=bool(args.load_model_strict))
        model = eval(args.wrapper_model)(args, model)
    return model
            
def prepare_model(model ,optimizer, lr_scheduler, criterion, loss_scaler, args):
    logsys = args.logsys
    if args.pretrain_weight and (torch.__version__[0] == "2" and args.torch_compile) and not args.continue_train:
        # if want to pretrain a model, then need load the model before torch.compile.
        only_model = ('fourcast' in args.mode) or (args.mode == 'finetune' and not args.continue_train)
        assert only_model
        load_model(model, path=args.pretrain_weight, only_model=only_model,loc='cpu', strict=bool(args.load_model_strict))
            
    if torch.__version__[0] == "2" and args.torch_compile:
        print(f"Now in torch 2.0, we use torch.compile")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)


    logsys.info(f'use lr_scheduler:{lr_scheduler}')
    args.pretrain_weight = args.pretrain_weight.strip()
    logsys.info(f"loading weight from {args.pretrain_weight}")
    # we put pretrain loading here due to we need load optimizer
    if args.torch_compile and args.pretrain_weight and not args.continue_train:
        start_epoch, start_step = 0, 0
        print(f"remind in torch compile mode, any pretrain model should be load before torch.compile and DistributedDataParallel")
    else:
        # if want to continue train a model, then need load the model after the torch.compile.
        start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, 
            optimizer, lr_scheduler, loss_scaler, path=args.pretrain_weight, 
            only_model= ('fourcast' in args.mode) or (args.mode=='finetune' and not args.continue_train),
            loc = 'cuda:{}'.format(args.gpu),strict=bool(args.load_model_strict))
        
    if not args.continue_train:min_loss = np.inf
    args.start_step = start_step
    args.min_loss   = min_loss

    start_epoch = start_epoch if args.continue_train else 0
    logsys.info(f"======> start from epoch:{start_epoch:3d}/{args.epochs:3d}")
    if args.more_epoch_train:
        assert args.pretrain_weight
        print(f"detect more epoch training, we will do a copy processing for {args.pretrain_weight}")
        os.system(f'cp {args.pretrain_weight} {args.pretrain_weight}-epoch{start_epoch}')
    logsys.info("done!")

                   
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=("FED" in args.model_type) or args.find_unused_parameters)
    else:
        model = model.cuda()

    if args.half_model:
        model = model.half()

    return model, optimizer, lr_scheduler, criterion, args

        
def build_model_and_optimizer(args):
    #cudnn.enabled         = True
    cudnn.benchmark = False  # will search a best CNN realized way at beginning
    cudnn.deterministic = True  # the key for continue training.
    logsys = args.logsys
    logsys.info(f"model args: img_size= {args.model.img_size}")
    logsys.info(f"model args: patch_size= {args.model.patch_size}")
    # ==============> Initial Model <=============
    if args.wrapper_model and 'Comb' in args.wrapper_model:
        model = build_combination_model(args)
    else:
        model = build_wrapper_model(args)
    logsys.info(f"use model ==> {model.__class__.__name__}")
    param_sum, buffer_sum, all_size = getModelSize(model)
    logsys.info(f"Rank: {args.rank}, Local_rank: {args.local_rank} | Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")

    optimizer,lr_scheduler,criterion = build_optimizer(args,model)
    loss_scaler                      = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    model, optimizer, lr_scheduler, criterion, loss_scaler, args = prepare_model(model, optimizer, lr_scheduler, criterion,loss_scaler, args)
    



    model.train_mode = args.mode
    model.random_time_step_train = args.random_time_step
    model.input_noise_std = args.input_noise_std
    model.history_length = args.history_length
    model.use_amp = bool(args.use_amp)
    model.clip_grad = args.clip_grad
    model.pred_len = args.pred_len
    model.accumulation_steps = args.accumulation_steps
    model.consistancy_alpha = deal_with_tuple_string(args.consistancy_alpha, [], dtype=float)
    model.consistancy_cut_grad = args.consistancy_cut_grad
    model.consistancy_eval = args.consistancy_eval
    if model.consistancy_eval:
        print(f'''
            setting model.consistancy_eval={model.consistancy_eval}, please make sure your model dont have running parameter like 
            BatchNorm, dropout>1 which will effect training.
        ''')
    model.vertical_constrain = args.vertical_constrain
    model.consistancy_activate_wall = args.consistancy_activate_wall
    model.mean_path_length = torch.zeros(1)
    model.wrapper_type = args.wrapper_model
    model.model_type = args.model_type

    if len(args.compute_graph) == 2:
        model.activate_stamps, model.activate_error_coef = args.compute_graph
        model.directly_esitimate_longterm_error = 0
    else:
        model.activate_stamps, model.activate_error_coef, model.directly_esitimate_longterm_error = args.compute_graph
        model.err_record = {}
        model.c1 = model.c2 = model.c3 = 1
    model.skip_constant_2D70N = args.skip_constant_2D70N
    if 'UVT' in args.wrapper_model:
        print(f"notice we are in property_pick mode, be careful. Current dataset is {args.dataset_type}")
        #assert "55" in args.dataset_flag
    if not hasattr(model, 'pred_channel_for_next_stamp') and args.input_channel != args.output_channel and args.output_channel == 68:
        model.pred_channel_for_next_stamp = list(range(0, 14*4-1)) + list(range(14*4, 69))
    return model, optimizer, lr_scheduler, criterion, loss_scaler

def build_optimizer(args, model):
    if args.opt == 'adamw':
        param_groups = timm.optim.optim_factory.param_groups_weight_decay(
            model, args.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, use_triton=False)
    elif args.opt == 'adamwbycase':
        from custom_optimizer import AdamWByCase
        optimizer = AdamWByCase([{'params': [p for name, p in model.named_parameters() if 'bias' in name],    'type':'tensor_adding'},
                                 {'params': [p for name, p in model.named_parameters(
                                 ) if 'bias' not in name], 'type':'tensor_contraction'}
                                 ], lr=args.lr, betas=(0.9, 0.95))
    elif args.opt == 'tiger':
        from custom_optimizer import Tiger
        optimizer = Tiger([{'params': [p for name, p in model.named_parameters() if 'bias' in name],    'type':'tensor_adding'},
                           {'params': [p for name, p in model.named_parameters(
                           ) if 'bias' not in name], 'type':'tensor_contraction'}
                           ], lr=args.lr)
    else:
        raise NotImplementedError
    GDMod_type = args.GDMod_type
    GDMod_lambda1 = args.GDMod_lambda1
    GDMod_lambda2 = args.GDMod_lambda2
    GDMode = {
        'NGmod_absolute': NGmod_absolute,
        'NGmod_delta_mean': NGmod_delta_mean,
        'NGmod_estimate_L2': NGmod_estimate_L2,
        'NGmod_absoluteNone': NGmod_absoluteNone,
        'NGmod_absolute_set_level': NGmod_absolute_set_level,
        'NGmod_RotationDeltaX': NGmod_RotationDeltaX,
        'NGmod_RotationDeltaE': NGmod_RotationDeltaE,
        'NGmod_RotationDeltaET': NGmod_RotationDeltaET,
        'NGmod_RotationDeltaETwo': NGmod_RotationDeltaETwo,
        'NGmod_RotationDeltaNmin': NGmod_RotationDeltaNmin,
        'NGmod_RotationDeltaESet': NGmod_RotationDeltaESet,
        'NGmod_RotationDeltaEThreeTwo': NGmod_RotationDeltaEThreeTwo,
        'NGmod_RotationDeltaXS': NGmod_RotationDeltaXS,
        'NGmod_RotationDeltaY': NGmod_RotationDeltaY,
        'NGmod_pathlength': NGmod_pathlength,

    }
    optimizer.grad_modifier = GDMode[GDMod_type](GDMod_lambda1, GDMod_lambda2,
                                                 sample_times=args.GDMod_sample_times,
                                                 L1_level=args.GDMod_L1_level, L2_level=args.GDMod_L2_level) if GDMod_type != 'off' else None
    if optimizer.grad_modifier is not None:
        optimizer.grad_modifier.ngmod_freq = args.ngmod_freq
        optimizer.grad_modifier.split_batch_chunk = args.split_batch_chunk
        optimizer.grad_modifier.update_mode = args.gmod_update_mode
        optimizer.grad_modifier.coef = None
        # bool(args.gdamp)# we will force two amp same
        optimizer.grad_modifier.use_amp = bool(args.use_amp)
        optimizer.grad_modifier.loss_wall = args.gd_loss_wall
        optimizer.grad_modifier.only_eval = args.gdeval
        optimizer.grad_modifier.gd_alpha = args.gd_alpha
        optimizer.grad_modifier.alpha_stratagy = args.gd_alpha_stratagy
        optimizer.grad_modifier.loss_target = args.gd_loss_target
        if args.gmod_coef:
            _, pixelnorm_std = np.load(args.gmod_coef)
            pixelnorm_std = torch.Tensor(pixelnorm_std).reshape(
                1, 70, 32, 64)  # <--- should pad
            assert not pixelnorm_std.isnan().any()
            assert not pixelnorm_std.isinf().any()
            # pixelnorm_std = torch.cat([pixelnorm_std[:,:55],
            #                torch.ones(1,1,32,64),
            #                pixelnorm_std[55:],
            #                torch.ones(1,1,32,64)],1
            #                )
            optimizer.grad_modifier.coef = pixelnorm_std
        optimizer.grad_modifier.path_length_regularize = args.path_length_regularize
        optimizer.grad_modifier.path_length_mode = args.path_length_mode if args.path_length_regularize else None
        optimizer.grad_modifier.rotation_regularize = args.rotation_regularize
        optimizer.grad_modifier.rotation_regular_mode = args.rotation_regular_mode if args.rotation_regular_mode else None
    lr_scheduler = None
    if args.sched:
        # if not args.scheduler_inital_epochs:
        #     args.scheduler_inital_epochs = args.epochs
        lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'pred_time_weighted_mse':
        print(args.dataset_patch_range)
        criterion = [CenterWeightMSE(args.dataset_patch_range - i, args.dataset_patch_range)
                     for i in range(args.time_step - args.history_length)]
    elif args.criterion == 'PressureWeightMSE':
        criterion = PressureWeightMSE()

    return optimizer, lr_scheduler, criterion

