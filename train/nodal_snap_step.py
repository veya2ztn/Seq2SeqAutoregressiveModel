
#########################################
######## nodal snap forward step #######
#########################################

def run_nodaloss_snap(model, batch, idxes, fourcastresult,dataset, property_select = [38,49],chunk_size=1024):
    time_step_1_mode=False
    L1meassures  = []
    L2meassures  = []
    L1meassure   = L2meassure = torch.zeros(1)
    start = batch[0:model.history_length] # start must be a list

    with torch.no_grad():
        for i in range(model.history_length,len(batch)):# i now is the target index
            func_model = lambda x:model(x)[:,property_select]
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,batch[i],dataset,time_step_1_mode)
                now = time.time()
                L1meassure = vmap(the_Nodal_L1_meassure(func_model), (0))(start[-1].unsqueeze(1)) # (B, Pick, W,H)
                L2meassure = vmap(the_Nodal_L2_meassure(func_model,chunk_size=chunk_size), (0))(start[-1].unsqueeze(1))# (B, Pick, W,H)
                print(f"step_{i:3d} L2 computing finish, cost:{time.time() - now }") 
                L1meassures.append(L1meassure.detach().cpu())
                L2meassures.append(L2meassure.detach().cpu())

    L1meassures = torch.stack(L1meassures,1) # (B, fourcast_num, Pick_property_num, W,H)
    L2meassures = torch.stack(L2meassures,1) # (B, fourcast_num, Pick_property_num, W,H)

    for idx, L1meassure,L2meassure in zip(idxes,L1meassures,L2meassures):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'L1meassure':L1meassure,"L2meassure":L2meassure}
    return fourcastresult

def snap_nodal_step(data_loader, model,logsys, property_select = [38,49],batch_limit=1,chunk_size=1024):
    model.eval()
    logsys.eval()
    status     = 'test'
    gpu        = dist.get_rank() if hasattr(model,'module') else 0
    Fethcher   = Datafetcher
    prefetcher = Fethcher(data_loader,next(model.parameters()).device)
    batches = len(data_loader)
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    device = next(model.parameters()).device
    data_cost = train_cost = rest_cost = 0
    now = time.time()
    model.clim = torch.Tensor(data_loader.dataset.clim_tensor).to(device)
    fourcastresult={}
    intervel = batches//100 + 1
    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            step           = inter_b.now
            idxes,batch    = prefetcher.next()
            batch          = make_data_regular(batch,half_model)
            fourcastresult = run_nodaloss_snap(model, batch, idxes, fourcastresult,data_loader.dataset, 
                                               property_select = property_select,chunk_size=chunk_size)
            if (step+1) % intervel==0 or step==0:
                outstring=(f"epoch:fourcast iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}]")
                inter_b.lwrite(outstring, end="\r")
            if inter_b.now > batch_limit:break
    return fourcastresult

def run_nodalosssnap(args, model,logsys,test_dataloader=None,property_select=[38,49]):
    import warnings
    warnings.filterwarnings("ignore")
    
    if test_dataloader is None:test_dataset,  test_dataloader = get_test_dataset(args)
    test_dataset = test_dataloader.dataset
    logsys.info_log_path = os.path.join(logsys.ckpt_root, f'nodal_snap_on_{test_dataset.split}_dataset.info')
    #args.force_fourcast=True
    gpu       = dist.get_rank() if hasattr(model,'module') else 0
    fourcastresult_path = os.path.join(logsys.ckpt_root,f"nodal_snap_on_{test_dataset.split}_dataset.gpu_{gpu}")

    if not os.path.exists(fourcastresult_path) or  args.force_fourcast:
        logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
        logsys.info("starting fourcast~!")
        fourcastresult  = snap_nodal_step(test_dataloader, model,logsys, property_select = property_select,
                                          batch_limit=args.batch_limit,chunk_size=args.chunk_size)
        fourcastresult['property_select'] = property_select
        torch.save(fourcastresult,fourcastresult_path)
        logsys.info(f"save fourcastresult at {fourcastresult_path}")
    else:
        logsys.info(f"load fourcastresult at {fourcastresult_path}")
        fourcastresult = torch.load(fourcastresult_path)

    if not args.distributed:
        create_nodal_loss_snap_metric_table(fourcastresult, logsys,test_dataset)
    
    return 1

def create_nodal_loss_snap_metric_table(fourcastresult, logsys,test_dataset):
    prefix_pool={
        'only_backward':"time_reverse_",
        'only_forward':""
    }
    prefix = prefix_pool[test_dataset.time_reverse_flag]

    if isinstance(fourcastresult,str):
        # then it is the fourcastresult path
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'nodal_snap_on_test_dataset.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val

    property_names = test_dataset.vnames
    property_select= fourcastresult['property_select']
    
    L1meassures = torch.stack([p['L1meassure'].cpu() for p in fourcastresult.values() if 'L1meassure' in p]) #(B, fourcast_num, Pick_property_num, W,H)
    L2meassures = torch.stack([p['L2meassure'].cpu() for p in fourcastresult.values() if 'L2meassure' in p]) #(B, fourcast_num, Pick_property_num, W,H)
    if len(L1meassures.shape)==6 and L1meassures.shape[2]==1:L1meassures=L1meassures[:,:,0]
    if len(L2meassures.shape)==6 and L2meassures.shape[2]==1:L2meassures=L2meassures[:,:,0]

    print(L1meassures.shape)
    print(L2meassures.shape)

    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        if torch.isnan(meassure).any() or torch.isinf(meassure).any():
            print(f"{metric_name}meassure has nan of inf")

    select_keys = [k for k in fourcastresult.keys() if isinstance(k,int)]

    print("create L1/L2 loss .................")
    # the first thing is to record the L1 measure per (B,P,W,H) per fourcast_num
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for fourcast_step, tensor_per_property in enumerate(meassure.permute(1,2,0,3,4).flatten(-3,-1)):
            for property_id, tensor in enumerate(tensor_per_property):
                property_name = property_names[property_select[property_id]]
                value = torch.mean((tensor - 1)**2)
                if torch.isnan(value):
                    print(f"{metric_name}_loss_for_{property_name}_with_{len(meassure)}_batches_on_{test_dataset.split}_at_{fourcast_step}_step_is bad")
                logsys.wandblog({f"{metric_name}_loss_for_{property_name}_with_{len(meassure)}_batches_on_{test_dataset.split}":value,'time_step':fourcast_step})     

    print("create mean std .................")
    # the first thing is to record the L1 measure per (B,P,W,H) per fourcast_num
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        table = []
        for fourcast_step, tensor_per_property in enumerate(meassure.permute(1,2,0,3,4).flatten(-3,-1)):
            for property_id, tensor in enumerate(tensor_per_property):
                property_name = property_names[property_select[property_id]]
                table.append([fourcast_step,property_name, idx, tensor.mean().item(),tensor.std().item()])
        logsys.add_table(f"{metric_name}_table", table , 0, ['fourcast_step',"property","idx","mean","std"])     

    print("create histgram .................")
    s_dir= os.path.join(logsys.ckpt_root,"figures")
    if not os.path.exists(s_dir):os.makedirs(s_dir)
    ## then we are going to plot the histgram
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for property_id, tensor_per_property in enumerate(meassure.permute(2,1,0,3,4).flatten(-3,-1)):
            property_name = property_names[property_select[property_id]]
            x_min = tensor_per_property.min()
            x_max = tensor_per_property.max()
            for fourcast_step, tensor in enumerate(tensor_per_property):
                
                data = tensor.numpy()
                name = f'{metric_name}_histogram_{property_name}_{len(meassure)}'
                if fourcast_step ==0:
                    table = wandb.Table(data=data[:,None], columns=[metric_name])
                    wandb.log({name+f"_at_time_step_{fourcast_step}": wandb.plot.histogram(table, metric_name,
                        title=f"{metric_name} histram for {property_name} with {len(meassure)} batches"),'time_step':fourcast_step})
                fig = plt.figure()
                smoothhist(data)
                plt.xlim(x_min,x_max)
                spath= os.path.join(s_dir,name+f'.step{fourcast_step}.png')
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:fig})
    ## then we are going to plot the heatmap
    s_dir= os.path.join(logsys.ckpt_root,"figures")
    if not os.path.exists(s_dir):os.makedirs(s_dir)
    for meassure, metric_name in zip([L1meassures,L2meassures],['L1','L2']):
        for property_id, tensor_per_property in enumerate(meassure.permute(2,1,0,3,4)):
            property_name = property_names[property_select[property_id]]
            x_min = 0
            x_max = 1.2
            for fourcast_step, tensor in enumerate(tensor_per_property):
                # tensor is (B, W, H), we only pick the first 
                the_map = tensor[0]
                #print(the_map.shape)
                start_time = select_keys[0]
                vmin = the_map.min()
                vmax = the_map.max()
                name = f"{metric_name}_map_{property_name}_start_from_{start_time}"
                spath= os.path.join(s_dir,name+f'.step{fourcast_step}.png')
                plt.imshow(the_map.numpy(),vmin=vmin,vmax=vmax,cmap='gray')
                plt.title(f"value range: {vmin:.3f}-{vmax:.3f}")
                plt.xticks([]);plt.yticks([])
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:wandb.Image(spath)})
                
    return
