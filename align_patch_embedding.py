from train.pretrain import *
from functools import partial
from model.align_model import PatchAlign_64_to_128


def get_train_and_valid_dataset(args,train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    dataset_type   = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    # we don't use valid processing
    # on the other hand, lets inject both shared tensor into dataset.
    train_dataset  = dataset_type(split="subtrain" if not args.debug else 'test',
                     dataset_tensor_1=train_dataset_tensor,record_load_tensor_1=train_record_load,
                     dataset_tensor_2=valid_dataset_tensor,record_load_tensor_2=valid_record_load,
                     **args.dataset_kargs)
    
    train_datasampler = DistributedSampler(train_dataset, shuffle=args.do_train_shuffle) if args.distributed else None
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader  = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=args.num_workers, pin_memory=True,
                                   drop_last=True,worker_init_fn=seed_worker,generator=g,shuffle=True if ((not args.distributed) and args.do_train_shuffle) else False)
    
    return   train_dataset,   None, train_dataloader, None


def run_one_iter(model, batch, criterion, status, gpu, dataset):
    iter_info_pool = {}
    assert len(batch) == 2  # the input is [B,P,64,128] and [B,P,128,256]
    low_level_embed_tensor, high_level_embed_tensor = model(batch[0], batch[1])
    loss = criterion(low_level_embed_tensor, high_level_embed_tensor)
    diff = loss
    return loss, diff, iter_info_pool, None, None


def run_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, logsys, status):

    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    # should be 16 for finetune. but I think its ok.
    accumulation_steps = model.accumulation_steps
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost = []
    train_cost = []
    rest_cost = []
    now = time.time()

    Fethcher = get_fetcher(status, data_loader)
    device = next(model.parameters()).device
    prefetcher = Fethcher(data_loader, device)
    # raise
    batches = len(data_loader)

    inter_b = logsys.create_progress_bar(
        batches, unit=' img', unit_scale=data_loader.batch_size)
    gpu = dist.get_rank() if hasattr(model, 'module') else 0

    if start_step == 0:
        optimizer.zero_grad()
    # intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    total_diff, total_num = torch.Tensor([0]).to(
        device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    path_loss = path_length = rotation_loss = None
    didunscale = False
    grad_modifier = optimizer.grad_modifier
    skip = False
    count_update = 0
    nan_detect = NanDetect(logsys, model.use_amp)

    while inter_b.update_step():
        # if inter_b.now>10:break
        step = inter_b.now
        batch = prefetcher.next()
        if step < start_step:
            continue
        batch = make_data_regular(batch, half_model)
        data_cost.append(time.time() - now)
        now = time.time()
        if status == 'train':
            if hasattr(model, 'set_step'):
                model.set_step(step=step, epoch=epoch)
            if hasattr(model, 'module') and hasattr(model.module, 'set_step'):
                model.module.set_step(step=step, epoch=epoch)
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                loss, abs_loss, iter_info_pool, _, _ = run_one_iter(
                    model, batch, criterion, 'train', gpu, data_loader.dataset)
            if nan_detect.nan_diagnose_weight(model, loss, loss_scaler):
                continue
            loss /= accumulation_steps
            loss_scaler.scale(loss).backward()
            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if (step+1) % accumulation_steps == 0:
                loss_scaler.step(optimizer)
                loss_scaler.update()
                count_update += 1
                optimizer.zero_grad()
                didunscale = False

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    loss, abs_loss, iter_info_pool, _, _ = run_one_iter(
                        model, batch, criterion, status, gpu, data_loader.dataset)

        if logsys.do_iter_log > 0:
            if logsys.do_iter_log == 1:
                iter_info_pool = {}  # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}'] = loss.item()
        else:
            iter_info_pool = {}

        total_diff += abs_loss.item()
        # total_num   += len(batch) - 1 #batch
        total_num += 1

        train_cost.append(time.time() - now)
        now = time.time()
        time_step_now = len(batch)
        if (step) % intervel == 0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches +
                              step, epoch_flag='iter')
        if (step) % intervel == 0 or step < 30:
            outstring = (
                f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            # print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")
        # if step>10:break

    if hasattr(model, 'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    loss_val = total_diff / total_num
    loss_val = loss_val.item()
    # torch.cuda.empty_cache()
    return loss_val


def build_model(args):
    # cudnn.enabled         = True
    cudnn.benchmark = False  # will search a best CNN realized way at beginning
    cudnn.deterministic = True  # the key for continue training.
    logsys = args.logsys
    logsys.info(f"model args: img_size= {args.img_size}")
    logsys.info(f"model args: patch_size= {args.patch_size}")
    args.model_kargs['unique_up_sample_channel'] = 0
    # ==============> Initial Model <=============
    if args.wrapper_model and 'Comb' in args.wrapper_model:
        args.model_kargs['history_length'] = 1
        assert args.model_type1
        assert args.model_type2
        args.model_kargs['in_chans'] = eval(
            args.wrapper_model).default_input_channel1
        args.model_kargs['out_chans'] = eval(
            args.wrapper_model).default_output_channel1
        # if args.model_type1 == 'AFNONet':
        #     pass
        # else:
        #     print("the ")
        #     #raise NotImplementedError
        backbone1 = eval(args.model_type1)(**args.model_kargs)
        args.model_kargs['in_chans'] = eval(
            args.wrapper_model).default_input_channel2
        args.model_kargs['out_chans'] = eval(
            args.wrapper_model).default_output_channel2

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
        model = eval(args.wrapper_model)(args, backbone1, backbone2,
                                         args.backbone1_ckpt_path, args.backbone2_ckpt_path)
    else:
        model = eval(args.model_type)(**args.model_kargs)
        if args.wrapper_model:

            if args.subweight:
                print(
                    f"in wrapper model, load subweight from {args.subweight}")
                # load_model(model.backbone,path=args.subweight,only_model=True, loc = 'cpu',strict=bool(args.load_model_strict))
                load_model(model, path=args.subweight, only_model=True,
                           loc='cpu', strict=bool(args.load_model_strict))
            model = eval(args.wrapper_model)(args, model)

    logsys.info(f"use model ==> {model.__class__.__name__}")
    local_rank = args.local_rank
    rank = args.rank
    if local_rank == 0:
        param_sum, buffer_sum, all_size = getModelSize(model)
        logsys.info(
            f"Rank: {args.rank}, Local_rank: {local_rank} | Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")
    if args.pretrain_weight and args.torch_compile and not args.continue_train:
        only_model = ('fourcast' in args.mode) or (
            args.mode == 'finetune' and not args.continue_train)
        assert only_model
        load_model(model, path=args.pretrain_weight, only_model=only_model,
                   loc='cpu', strict=bool(args.load_model_strict))
    if torch.__version__[0] == "2" and args.torch_compile:
        print(f"Now in torch 2.0, we use torch.compile")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                          args.gpu], find_unused_parameters=("FED" in args.model_type) or args.find_unused_parameters)
    else:
        model = model.cuda()

    if args.half_model:
        model = model.half()

    model.train_mode = args.mode

    model.random_time_step_train = args.random_time_step
    model.input_noise_std = args.input_noise_std
    model.history_length = args.history_length
    model.use_amp = bool(args.use_amp)
    model.clip_grad = args.clip_grad
    model.pred_len = args.pred_len
    model.accumulation_steps = args.accumulation_steps
    model.consistancy_alpha = deal_with_tuple_string(
        args.consistancy_alpha, [], dtype=float)
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
    compute_graph = parser_compute_graph(args.compute_graph_set)
    if len(compute_graph) == 2:
        model.activate_stamps, model.activate_error_coef = compute_graph
        model.directly_esitimate_longterm_error = 0
    else:
        model.activate_stamps, model.activate_error_coef, model.directly_esitimate_longterm_error = compute_graph
        model.err_record = {}
        model.c1 = model.c2 = model.c3 = 1
    model.skip_constant_2D70N = args.skip_constant_2D70N
    if 'UVT' in args.wrapper_model:
        print(
            f"notice we are in property_pick mode, be careful. Current dataset is {args.dataset_type}")
        # assert "55" in args.dataset_flag
    if not hasattr(model, 'pred_channel_for_next_stamp') and args.input_channel != args.output_channel and args.output_channel == 68:
        model.pred_channel_for_next_stamp = list(
            range(0, 14*4-1)) + list(range(14*4, 69))
    return model
def main_worker(local_rank, ngpus_per_node, args, result_tensor=None,
                train_dataset_tensor=None, train_record_load=None, valid_dataset_tensor=None, valid_record_load=None):
    if local_rank == 0:
        print(f"we are at mode={args.mode}")
    ##### locate the checkpoint dir ###########
    args.gpu = args.local_rank = gpu = local_rank
    ##### parse args: dataset_kargs / model_kargs / train_kargs  ###########
    args = parse_default_args(args)
    SAVE_PATH = get_ckpt_path(args)
    args.SAVE_PATH = str(SAVE_PATH)
    ########## inital log ###################
    logsys = create_logsys(args)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        logsys.info(
            f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    model = build_model(args)
    # param_groups    = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer, lr_scheduler, criterion = build_optimizer(args, model)
    loss_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    logsys.info(f'use lr_scheduler:{lr_scheduler}')

    args.pretrain_weight = args.pretrain_weight.strip()
    logsys.info(f"loading weight from {args.pretrain_weight}")
    # we put pretrain loading here due to we need load optimizer
    if args.torch_compile and args.pretrain_weight and not args.continue_train:
        start_epoch, start_step, min_loss = 0, 0, 0
        print(f"remind in torch compile mode, any pretrain model should be load before torch.compile and DistributedDataParallel")
    else:
        start_epoch, start_step, min_loss = load_model(model.module if args.distributed else model, optimizer, lr_scheduler, loss_scaler, path=args.pretrain_weight,
                                                       only_model=('fourcast' in args.mode) or (args.mode == 'finetune' and not args.continue_train), loc='cuda:{}'.format(args.gpu), strict=bool(args.load_model_strict))
    start_epoch = start_epoch if args.continue_train else 0
    logsys.info(f"======> start from epoch:{start_epoch:3d}/{args.epochs:3d}")
    if args.more_epoch_train:
        assert args.pretrain_weight
        print(
            f"detect more epoch training, we will do a copy processing for {args.pretrain_weight}")
        os.system(
            f'cp {args.pretrain_weight} {args.pretrain_weight}-epoch{start_epoch}')
    logsys.info("done!")

    # =======================> start training <==========================
    logsys.info(
        f"entering {args.mode} training in {next(model.parameters()).device}")
    now_best_path = SAVE_PATH / 'backbone.best.pt'
    latest_ckpt_p = SAVE_PATH / 'pretrain_latest.pt'
    now_Z500_path = SAVE_PATH / 'fourcast.best.pt'
    test_dataloader = None
    train_loss = -1
    Z500_now = Z500_best = -1

    train_dataset, val_dataset, train_dataloader, val_dataloader = get_train_and_valid_dataset(args,
                                                                                               train_dataset_tensor=train_dataset_tensor, train_record_load=train_record_load,
                                                                                               valid_dataset_tensor=valid_dataset_tensor, valid_record_load=valid_record_load)
    logsys.info(f"use dataset ==> {train_dataset.__class__.__name__}")
    logsys.info(f"Start training for {args.epochs} epochs")
    master_bar = logsys.create_master_bar(args.epochs)
    accu_list = ['valid_loss']
    metric_dict = logsys.initial_metric_dict(accu_list)
    banner = logsys.banner_initial(args.epochs, args.SAVE_PATH)
    logsys.banner_show(0, args.SAVE_PATH)
    val_loss = 1.234
    train_loss = -1
    if args.tracemodel:
        logsys.wandb_watch(model, log_freq=100)
    for epoch in master_bar:
        if epoch < start_epoch:
            continue
        # do fourcast once at begining
        fast_set_model_epoch(model, epoch=epoch,
                             epoch_total=args.epochs, eval_mode=False)
        logsys.record(
            'learning rate', optimizer.param_groups[0]['lr'], epoch, epoch_flag='epoch')
        train_loss = run_one_epoch(epoch, start_step, model, criterion,
                                   train_dataloader, optimizer, loss_scaler, logsys, 'train')
        freeze_learning_rate = (args.scheduler_min_lr and optimizer.param_groups[0]['lr'] < args.scheduler_min_lr) and (
            args.scheduler_inital_epochs and epoch > args.scheduler_inital_epochs)
        if (not args.more_epoch_train) and (lr_scheduler is not None) and not freeze_learning_rate:
            lr_scheduler.step(epoch)
        if args.valid_every_epoch and ((epoch % args.valid_every_epoch == 0) or (epoch == args.epochs - 1)):
            fast_set_model_epoch(model, epoch=epoch,
                                 epoch_total=args.epochs, eval_mode=True)
            val_loss = run_one_epoch(epoch, start_step, model, criterion,
                                     val_dataloader, optimizer, loss_scaler, logsys, 'valid')

        logsys.metric_dict.update({'valid_loss': val_loss}, epoch)
        logsys.banner_show(epoch, args.SAVE_PATH, train_losses=[train_loss])
        if (not args.distributed) or (args.rank == 0 and local_rank == 0):
            logsys.info(
                f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}", show=False)
            logsys.record('train', train_loss, epoch, epoch_flag='epoch')
            logsys.record('valid', val_loss, epoch, epoch_flag='epoch')
            if val_loss < min_loss:
                min_loss = val_loss
                if epoch > args.epochs//10:
                    logsys.info(f"saving best model ....", show=False)
                    save_model(model, path=now_best_path, only_model=True)
                    logsys.info(f"done;", show=False)
                # if last_best_path is not None:os.system(f"rm {last_best_path}")
                # last_best_path= now_best_path
                logsys.info(f"The best accu is {val_loss}", show=False)
            logsys.record('best_loss', min_loss, epoch, epoch_flag='epoch')
            update_experiment_info(experiment_hub_path, epoch, args)
            if ((epoch >= args.save_warm_up) and (epoch % args.save_every_epoch == 0)) or (epoch == args.epochs-1) or (epoch in args.epoch_save_list):
                logsys.info(f"saving latest model ....", show=False)
                save_model(model, epoch=epoch+1, step=0, optimizer=optimizer, lr_scheduler=lr_scheduler,
                           loss_scaler=loss_scaler, min_loss=min_loss, path=latest_ckpt_p)
                logsys.info(f"done ....", show=False)
                if epoch in args.epoch_save_list:
                    save_model(
                        model, path=f'{latest_ckpt_p}-epoch{epoch}', only_model=True)
                    # os.system(f'cp {latest_ckpt_p} {latest_ckpt_p}-epoch{epoch}')
    # and not args.distributed:
    if os.path.exists(now_best_path) and args.do_final_fourcast:
        if not isinstance(args.do_final_fourcast, str):
            args.do_final_fourcast = 'backbone.best.pt'
        # <--this is not safe, but fine.
        now_best_path = SAVE_PATH / args.do_final_fourcast
        logsys.info(
            f"we finish training, then start test on the best checkpoint {now_best_path}")
        start_epoch, start_step, min_loss = load_model(
            model.module if args.distributed else model, path=now_best_path, only_model=True, loc='cuda:{}'.format(args.gpu))
        run_fourcast(args, model, logsys)
    if result_tensor is not None and local_rank == 0:
        result_tensor[local_rank] = min_loss
    logsys.close()


def main(args=None):
    if args is None:
        args = get_args()
    args = distributed_initial(args)
    train_dataset_tensor, valid_dataset_tensor, train_record_load, valid_record_load = create_memory_templete(
        args)
    result_tensor = torch.zeros(1).share_memory_()
    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = args.ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args, result_tensor,
                                    train_dataset_tensor, train_record_load,
                                    valid_dataset_tensor, valid_record_load))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, args.ngpus_per_node, args, result_tensor,
                    train_dataset_tensor, train_record_load, valid_dataset_tensor, valid_record_load)
    return result_tensor


if __name__ == '__main__':
    main()
