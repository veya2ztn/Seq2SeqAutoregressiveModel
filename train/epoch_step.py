
def get_fetcher(status,data_loader):
    if (status =='train' and \
        data_loader.dataset.use_offline_data and \
        data_loader.dataset.split=='train' and \
        'Patch' in data_loader.dataset.__class__.__name__):
      return RandomSelectPatchFetcher 
    elif (status =='train' and 'Multibranch' in data_loader.dataset.__class__.__name__):
        # notice we should not do valid when use this fetcher
        return RandomSelectMultiBranchFetcher
    else:
        return Datafetcher

def run_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    if optimizer.grad_modifier and optimizer.grad_modifier.__class__.__name__=='NGmod_RotationDeltaEThreeTwo':
        assert data_loader.dataset.time_step==3
        return run_one_epoch_three2two(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status)
    else:
        return run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status)

def run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    
    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    accumulation_steps = model.accumulation_steps # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost  = []
    train_cost = []
    rest_cost  = []
    now = time.time()

    Fethcher   = get_fetcher(status,data_loader)
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    #raise
    batches    = len(data_loader)

    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    #intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    path_loss = path_length = rotation_loss = None
    didunscale = False
    grad_modifier = optimizer.grad_modifier
    skip = False
    count_update = 0
    nan_detect   = NanDetect(logsys,model.use_amp)

    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        
        run_gmod = False
        if grad_modifier is not None:
            control = step if grad_modifier.update_mode==2 else count_update
            run_gmod = (control%grad_modifier.ngmod_freq==0)

        batch = prefetcher.next()

        # In this version(2022-12-22) we will split normal and ngmod processing
        # we will do normal train with normal batchsize and learning rate multitimes
        # then do ngmod train 
        # Notice, one step = once backward not once forward
        
        #[print(t[0].shape) for t in batch]
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now);now = time.time()
        if status == 'train':
            if hasattr(model, 'set_step'):
                model.set_step(step=step, epoch=epoch, step_total=batches)
            if hasattr(model, 'module') and hasattr(model.module, 'set_step'):
                model.module.set_step(
                    step=step, epoch=epoch, step_total=batches)
            # if model.train_mode =='pretrain':
            #     time_truncate = max(min(epoch//3,data_loader.dataset.time_step),2)
            #     batch=batch[:model.history_length -1 + time_truncate]

            # the normal initial method will cause numerial explore by using timestep > 4 senenrio.
            
            if grad_modifier is not None and run_gmod and (grad_modifier.update_mode==2):
                chunk = grad_modifier.split_batch_chunk
                ng_accu_times = max(data_loader.batch_size//chunk,1)

                batch_data_full = batch[0]
                
                ## nodal loss
                #### to avoid overcount,
                ## use model.module rather than model in Distribution mode is fine.
                # It works, although I think it is not safe. 
                # use model in distribution mode will go wrong, altough it can work in old code version.
                # I suppose it is related to the graph optimization processing in pytorch.
                for chunk_id in range(ng_accu_times):
                    if isinstance(batch_data_full,list):
                        batch_data = torch.cat([ttt[chunk_id*chunk:(chunk_id+1)*chunk].flatten(1,-1) for ttt in batch_data_full],1)
                    else:
                        batch_data = batch_data_full[chunk_id*chunk:(chunk_id+1)*chunk]
                    ngloss=None
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        if grad_modifier.lambda1!=0:
                            if ngloss is None:ngloss=0
                            Nodeloss1 = grad_modifier.getL1loss(model.module if hasattr(model,'module') else model, batch_data,coef=grad_modifier.coef)/ng_accu_times
                            ngloss  += grad_modifier.lambda1 * Nodeloss1
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            if ngloss is None:ngloss=0
                            Nodeloss2 = grad_modifier.getL2loss(model.module if hasattr(model,'module') else model, batch_data,coef=grad_modifier.coef)/ng_accu_times
                            ngloss += grad_modifier.lambda2 * Nodeloss2
                            Nodeloss2=Nodeloss2.item()
                    if ngloss is not None:
                        loss_scaler.scale(ngloss).backward()    
                        # if model.use_amp:
                        #     loss_scaler.scale(ngloss).backward()    
                        # else:
                        #     ngloss.backward()

                    # for idx,(name,p) in enumerate(model.named_parameters()):
                    #     print(f"{chunk_id}:{name}:{p.device}:{p.norm()}:{p.grad.norm() if p.grad is not None else None}")
                    #     if idx>10:break
                    # print("===========================")
                #raise


            with torch.cuda.amp.autocast(enabled=model.use_amp):
                loss, abs_loss, iter_info_pool,ltmv_pred,target  =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
                ## nodal loss
                if (grad_modifier is not None) and (run_gmod) and (grad_modifier.update_mode==2):
                    if grad_modifier.lambda1!=0:
                        Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                        loss += grad_modifier.lambda1 * Nodeloss1
                        Nodeloss1=Nodeloss1.item()
                    if grad_modifier.lambda2!=0:
                        Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                        if Nodeloss2>0:
                            loss += grad_modifier.lambda2 * Nodeloss2
                        Nodeloss2=Nodeloss2.item()
        
            if nan_detect.nan_diagnose_weight(model,loss,loss_scaler):continue
            
            loss /= accumulation_steps
            loss_scaler.scale(loss).backward()    
            # else:
            #     loss.backward()

            #select_para= list(range(5)) + [-5,-4,-3,-2,-1] 
             
            #pnormlist = [[name,p.grad.norm()] for name, p in model.named_parameters() if p.grad is not None]
            #for i in select_para:
            #    name,norm = pnormlist[i]
            #    print(f'before:gpu:{device} - {name} - {norm}')
            
            path_loss = path_length = None
            if grad_modifier and grad_modifier.path_length_regularize and step%grad_modifier.path_length_regularize==0:
                mean_path_length = model.mean_path_length.to(device)
                
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    path_loss, mean_path_length, path_lengths = grad_modifier.getPathLengthloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], mean_path_length, path_length_mode=grad_modifier.path_length_mode )
                
                if path_loss > grad_modifier.loss_wall:
                    the_loss = path_loss*grad_modifier.gd_alpha
                    loss_scaler.scale(the_loss).backward()    
                # if grad_modifier.use_amp:
                #     loss_scaler.scale(the_loss).backward()    
                # else:
                #     the_loss.backward()

                #pnormlist = [[name,p.grad] for name, p in model.named_parameters() if p.grad is not None]
                #name,norm = pnormlist[0]
                #print(f'before:gpu:{device} - {name} - {norm}')
                #for i in select_para:
                #    name,norm = pnormlist[i]
                #    print(f'before:gpu:{device} - {name} - {norm}')

                if hasattr(model,'module'):
                    mean_path_length = mean_path_length/dist.get_world_size()
                    # dist.barrier()# <--- its doesn't matter
                    dist.all_reduce(mean_path_length)
                    
                #pnormlist = [[name,p.grad] for name, p in model.named_parameters() if p.grad is not None]
                #name,norm = pnormlist[0]
                #print(f'before:gpu:{device} - {name} - {norm}')
                #for i in select_para:
                #    name,norm = pnormlist[i]
                #    print(f'after:gpu:{device} - {name} - {norm}')
                   
                model.mean_path_length = mean_path_length.detach().cpu()
                path_loss = path_loss.item()
                path_lengths=path_lengths.mean().item()

                
            rotation_loss = None
            if grad_modifier and grad_modifier.rotation_regularize and step%grad_modifier.rotation_regularize==0:
                # amp will destroy the train
                #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
                #rotation_loss.backward()
                if grad_modifier.only_eval:
                    with torch.no_grad(): 
                        with torch.cuda.amp.autocast(enabled=model.use_amp):
                            rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, 
                            batch[0], ltmv_pred.detach() if len(batch)==2 else None, batch[1],
                            rotation_regular_mode = grad_modifier.rotation_regular_mode)                     
                else:
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], ltmv_pred.detach() if len(batch)==2 else None, batch[1],#target,##<-- same reason, should be next stamp rather than last
                                rotation_regular_mode = grad_modifier.rotation_regular_mode)                    
                    # notice this y must be x_{t+1}_pred, this works when time_step=2,
                    # when time_step > 2, the provided ltmv_pred is the last pred not the next.
                    if grad_modifier.alpha_stratagy == 'softwarmup50.90':
                        gd_alpha = grad_modifier.gd_alpha*min(max((np.exp((epoch-50)/40)-1)/(np.exp(1)-1),0),1)
                    elif grad_modifier.alpha_stratagy == 'softwarmup00.80':
                        gd_alpha = grad_modifier.gd_alpha*min(max((np.exp((epoch-0)/80)-1)/(np.exp(1)-1),0),1)
                    elif grad_modifier.alpha_stratagy == 'normal':
                        gd_alpha=grad_modifier.gd_alpha
                    else:
                        raise NotImplementedError

                    if (rotation_loss > grad_modifier.loss_wall) and gd_alpha>0: #default grad_modifier.loss_wall is 0
                        if grad_modifier.loss_target:
                            the_loss = abs(rotation_loss-grad_modifier.loss_target)*gd_alpha
                        else:
                            the_loss = rotation_loss*gd_alpha
                        loss_scaler.scale(the_loss).backward() 
                    # if grad_modifier.use_amp:
                    #     loss_scaler.scale(the_loss).backward()    
                    # else:
                    #     the_loss.backward()
                
                rotation_loss = rotation_loss.item()
                
            # In order to use multiGPU train, I have to use Loss update scenario, suprisely, it is not worse than split scenario
            # if optimizer.grad_modifier is not None:
            #     #assert not model.use_amp
            #     #assert accumulation_steps == 1 
            #     if model.use_amp and not didunscale:
            #         loss_scaler.unscale_(optimizer) # do unscaler here for right gradient modify like clip or norm
            #         didunscale = True
            #     assert len(batch)==2 # we now only allow one 
            #     assert isinstance(batch[0],torch.Tensor)
            #     with controlamp(model.use_amp)():
            #         optimizer.grad_modifier.backward(model, batch[0], batch[1], strict=False)
            #         Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
            
            # if nan_detect.nan_diagnose_grad(model,loss,loss_scaler):
            #     optimizer.zero_grad()
            #     continue

            if hasattr(model,'module') and grad_modifier:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/dist.get_world_size()
                        dist.all_reduce(p.grad) #<--- pytorch DDP doesn't support high order gradient. This step need!

            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if model.directly_esitimate_longterm_error and 'runtime' in model.directly_esitimate_longterm_error:
                assert 'logoffset' in model.directly_esitimate_longterm_error
                normlized_type = normlized_coef_type_bonded
                for key in model.err_record.keys():
                    if hasattr(model, 'module')  :
                        dist.barrier()
                        dist.all_reduce(model.err_record[key])
                    model.err_record[key] = model.err_record[key][None]
                c1,c2,c3 = compute_coef(model.err_record , model.directly_esitimate_longterm_error, normlized_type)
                if not hasattr(model,'clist_buffer'):model.clist_buffer={'c1':[],'c2':[],'c3':[]}
                for name,c in zip(['c1','c2','c3'],[c1.item(),c2.item(),c3.item()]):
                    model.clist_buffer[name].append(c)
                    if len(model.clist_buffer[name])>100:
                        model.clist_buffer[name].pop(0)
                        setattr(model,name,np.mean(model.clist_buffer[name]))
                # model.c1 = c1;model.c2 = c2;model.c3 = c3
            
            if (step+1) % accumulation_steps == 0:
                loss_scaler.step(optimizer)
                loss_scaler.update()   
                # if model.use_amp:                  
                #     loss_scaler.step(optimizer)
                #     loss_scaler.update()   
                # else:
                #     optimizer.step()
                count_update += 1
                optimizer.zero_grad()
                didunscale = False
        
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    loss, abs_loss, iter_info_pool,ltmv_pred,target =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
                    if optimizer.grad_modifier is not None:
                        if grad_modifier.lambda1!=0:
                            Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss2=Nodeloss2.item()

        if logsys.do_iter_log > 0:
            if logsys.do_iter_log ==  1:iter_info_pool={} # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}']       =  loss.item()
        else:
            iter_info_pool={}
        if Nodeloss1  > 0:iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}']  = Nodeloss1
        if Nodeloss2  > 0:iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}']  = Nodeloss2
        if path_loss is not None:iter_info_pool[f'{status}_path_loss_gpu{gpu}']  = path_loss
        if path_length is not None:iter_info_pool[f'{status}_path_length_gpu{gpu}']  = path_length
        if rotation_loss is not None:iter_info_pool[f'{status}_rotation_loss_gpu{gpu}']= rotation_loss
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            #print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost  = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")
        #if step>10:break


    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    if model.directly_esitimate_longterm_error and 'during_valid' in model.directly_esitimate_longterm_error and status == 'valid':
        normlized_type = normlized_coef_type2
        if "needbase" in model.directly_esitimate_longterm_error:normlized_type = normlized_coef_type3
        elif "vallina" in model.directly_esitimate_longterm_error:normlized_type = normlized_coef_type0
        if 'logoffset' in model.directly_esitimate_longterm_error:
            normlized_type = normlized_coef_type_bonded
        if 'per_feature' in model.directly_esitimate_longterm_error:
            for key in model.err_record.keys():
                model.err_record[key] = torch.cat(model.err_record[key]).mean(0)
                if hasattr(model, 'module')  :
                    dist.barrier()
                    dist.all_reduce(model.err_record[key])
                model.err_record[key] = model.err_record[key]
            c1,c2,c3 = compute_coef(model.err_record , model.directly_esitimate_longterm_error, normlized_type)
        elif 'per_sample' in model.directly_esitimate_longterm_error:
            for key in model.err_record.keys():
                model.err_record[key] = torch.cat(model.err_record[key])# (B,)
            c1,c2,c3 = compute_coef(model.err_record , model.directly_esitimate_longterm_error, normlized_type)
            #print(f"===> before <=== gpu:{device} c1={c1:.4f} c2={c2:.4f} c3={c3:.4f}")
            if hasattr(model, 'module'):
                for x in [c1, c2, c3]:
                    dist.barrier()
                    dist.all_reduce(x)
            #print(f"===> after <=== gpu:{device} c1={c1:.4f} c2={c2:.4f} c3={c3:.4f}")
        else:
            raise NotImplementedError
        model.c1 = c1;model.c2 = c2;model.c3 = c3
        model.err_record = {}
        #print(c1,c2,c3)
        #raise

    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    #torch.cuda.empty_cache()
    return loss_val

def compute_coef(err_record, flag,normlized_type):
    # will record mse error no matter we use log loss or other loss
    if 'valid' in flag:
        e1   =(err_record[0,1,1]+err_record[0,1,2] + err_record[0,1,3])/3
        a0   = err_record[1,2,2]/err_record[0,1,2]
        a1   = err_record[1,3,3]/err_record[0,1,3]
        e2   = err_record[0,2,2]
        e3   = err_record[0,3,3]
    else: # in train mode
        e1   = err_record[0,1,1]
        e2   = err_record[0,2,2]
        e3   = err_record[0,3,3]
        a0   = (e2 - e1)/e1
        a1   = (e3 - e1)/e2
    c1_l,c2_l,c3_l = [],[],[]
    if isinstance(e1, float):
        e1,a0,a1,e1,e2,e3 = [e1],[a0],[a1],[e1],[e2],[e3]
    for _e1,_a0,_a1,_e1,_e2,_e3 in zip(e1,a0,a1,e1,e2,e3):
        _e1,_e2,_e3 = float(_e1), float(_e2), float(_e3)
        _a0,_a1   = float(_a0), float(_a1)
        if _a0 > 0.9:continue
        if _a1 > 0.9:continue
        c1,c2,c3 = calculate_coef(_e1,_a0,_a1,rank=int(flag.split('_')[-1]))
        #print(f"e1:{_e1:.4f} e2:{_e2:.4f} e3:{_e3:.4f} c1:{c1:.4f} c2:{c2:.4f} c3:{c3:.4f}")
        c1,c2,c3 = normlized_type(c1,c2,c3,_e1,_e2,_e3)
        #print(f"e1:{_e1:.4f} e2:{_e2:.4f} e3:{_e3:.4f} c1:{c1:.4f} c2:{c2:.4f} c3:{c3:.4f}")
        #print("====================")
        c1_l.append(c1);c2_l.append(c2);c3_l.append(c3)
    c1 = torch.Tensor(c1_l).to(e1.device).mean()
    c2 = torch.Tensor(c2_l).to(e1.device).mean()
    c3 = torch.Tensor(c3_l).to(e1.device).mean()
    return c1,c2,c3

def run_one_epoch_three2two(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    
    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    accumulation_steps = model.accumulation_steps # should be 16 for finetune. but I think its ok.
    half_model = next(model.parameters()).dtype == torch.float16

    data_cost  = []
    train_cost = []
    rest_cost  = []
    now = time.time()

    
    Fethcher   = get_fetcher(status,data_loader)
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    #raise
    batches    = len(data_loader)

    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    gpu        = dist.get_rank() if hasattr(model,'module') else 0

    if start_step == 0:optimizer.zero_grad()
    #intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    nan_count = 0
    Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    path_loss = path_length = rotation_loss = None
    didunscale = False
    grad_modifier = optimizer.grad_modifier
    skip = False
    count_update = 0
    
    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        
        run_gmod = False
        if grad_modifier is not None:
            control = step if grad_modifier.update_mode==2 else count_update
            run_gmod = (control%grad_modifier.ngmod_freq==0)

        batch = prefetcher.next()

        # In this version(2022-12-22) we will split normal and ngmod processing
        # we will do normal train with normal batchsize and learning rate multitimes
        # then do ngmod train 
        # Notice, one step = once backward not once forward
        
        #[print(t[0].shape) for t in batch]
        if step < start_step:continue
        #batch = data_loader.dataset.do_normlize_data(batch)
        
        batch = make_data_regular(batch,half_model)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now);now = time.time()
        assert len(batch) == 3
        if status == 'train':
            if hasattr(model,'set_step'):model.set_step(step=step,epoch=epoch)
            if hasattr(model,'module') and hasattr(model.module,'set_step'):model.module.set_step(step=step,epoch=epoch)
            
            # one batch is [(B,P,W,H),(B,P,W,H),(B,P,W,H)]
            # the the input should be 
            batch = [torch.cat([batch[0],batch[1]]),torch.cat([batch[1],batch[2]])]
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                loss, abs_loss, iter_info_pool,ltmv_pred,target  =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            
            loss, nan_count, skip = nan_diagnose_weight(model,loss,nan_count,logsys)
            if skip:continue
            
            loss /= accumulation_steps
            loss_scaler.scale(loss).backward()    

            # if model.use_amp:
            #     loss_scaler.scale(loss).backward()    
            # else:
            #     loss.backward()
  
            rotation_loss = None
            if grad_modifier and grad_modifier.rotation_regularize and step%grad_modifier.rotation_regularize==0:
                # amp will destroy the train
                #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
                #rotation_loss.backward()
                if grad_modifier.only_eval:
                    with torch.no_grad(): 
                        with torch.cuda.amp.autocast(enabled=model.use_amp):
                            rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, 
                                    batch[0], 
                                    ltmv_pred.detach(),
                                    target,
                                    rotation_regular_mode = grad_modifier.rotation_regular_mode)                     
                else:
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
                                batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)                    
                    the_loss = rotation_loss*grad_modifier.gd_alpha
                    if grad_modifier.use_amp:
                        loss_scaler.scale(the_loss).backward()    
                    else:
                        the_loss.backward()
                        
                rotation_loss = rotation_loss.item()


            # In order to use multiGPU train, I have to use Loss update scenario, suprisely, it is not worse than split scenario
            # if optimizer.grad_modifier is not None:
            #     #assert not model.use_amp
            #     #assert accumulation_steps == 1 
            #     if model.use_amp and not didunscale:
            #         loss_scaler.unscale_(optimizer) # do unscaler here for right gradient modify like clip or norm
            #         didunscale = True
            #     assert len(batch)==2 # we now only allow one 
            #     assert isinstance(batch[0],torch.Tensor)
            #     with controlamp(model.use_amp)():
            #         optimizer.grad_modifier.backward(model, batch[0], batch[1], strict=False)
            #         Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
            

            #nan_count, skip = nan_diagnose_grad(model,nan_count,logsys)
            # if skip:
            #     optimizer.zero_grad()
            #     continue
            if hasattr(model,'module') and grad_modifier:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/dist.get_world_size()
                        dist.all_reduce(p.grad) #<--- pytorch DDP doesn't support high order gradient. This step need!

            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if (step+1) % accumulation_steps == 0:
                loss_scaler.step(optimizer)
                loss_scaler.update()   
                # if model.use_amp:                  
                #     loss_scaler.step(optimizer)
                #     loss_scaler.update()   
                # else:
                #     optimizer.step()
                count_update += 1
                optimizer.zero_grad()
                didunscale = False
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    loss, abs_loss, iter_info_pool,ltmv_pred,target =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
                    if optimizer.grad_modifier is not None:
                        if grad_modifier.lambda1!=0:
                            Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss1=Nodeloss1.item()
                        if grad_modifier.lambda2!=0:
                            Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
                            Nodeloss2=Nodeloss2.item()
        if logsys.do_iter_log > 0:
            if logsys.do_iter_log ==  1:iter_info_pool={} # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}']       =  loss.item()
        else:
            iter_info_pool={}
        if Nodeloss1  > 0:iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}']  = Nodeloss1
        if Nodeloss2  > 0:iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}']  = Nodeloss2
        if path_loss is not None:iter_info_pool[f'{status}_path_loss_gpu{gpu}']  = path_loss
        if path_length is not None:iter_info_pool[f'{status}_path_length_gpu{gpu}']  = path_length
        if rotation_loss is not None:iter_info_pool[f'{status}_rotation_loss_gpu{gpu}']= rotation_loss
        total_diff  += abs_loss.item()
        #total_num   += len(batch) - 1 #batch 
        total_num   += 1 

        train_cost.append(time.time() - now);now = time.time()
        time_step_now = len(batch)
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            #print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost  = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")


    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    return loss_val

