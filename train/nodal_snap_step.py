import torch

#########################################
######## nodal snap forward step #######
#########################################


def run_one_epoch_nodal(epoch, start_step, model, criterion, data_loader,
                         optimizer, loss_scaler, logsys, status, train_config, grad_modifier):

    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError

    Fethcher = get_fetcher(status, data_loader)
    device = next(model.parameters()).device
    prefetcher = Fethcher(data_loader, device)

    data_cost = []
    train_cost = []
    rest_cost = []
    now = time.time()

    batches = len(data_loader)
    inter_b = logsys.create_progress_bar(
        batches, unit=' img', unit_scale=data_loader.batch_size)
    gpu = get_local_rank()

    # should be 16 for finetune. but I think its ok.
    accumulation_steps = train_config.accumulation_steps

    if start_step == 0:
        optimizer.zero_grad()
    #intervel = batches//100 + 1
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    total_diff, total_num = torch.Tensor([0]).to(
        device), torch.Tensor([0]).to(device)
    Nodeloss1 = Nodeloss2 = 0
    path_loss = path_length = rotation_loss = None
    count_update = 0
    nan_detect = NanDetect(logsys, model.use_amp)

    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        batch = prefetcher.next()
        if step < start_step:
            continue

        run_gmod = False
        if grad_modifier is not None:
            control = step if grad_modifier.update_mode == 2 else count_update
            run_gmod = (control % grad_modifier.ngmod_freq == 0)

        # In this version(2022-12-22) we will split normal and ngmod processing
        # we will do normal train with normal batchsize and learning rate multitimes
        # then do ngmod train
        # Notice, one step = once backward not once forward

        #[print(t[0].shape) for t in batch]

        #batch = data_loader.dataset.do_normlize_data(batch)

        batch = make_data_regular(batch)
        #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
        data_cost.append(time.time() - now)
        now = time.time()
        if status == 'train':
            apply_model_step(model, step=step, epoch=epoch, step_total=batches)

            # the normal initial method will cause numerial explore by using timestep > 4 senenrio.

            if grad_modifier is not None and run_gmod and (grad_modifier.update_mode == 2):
                chunk = grad_modifier.split_batch_chunk
                ng_accu_times = max(data_loader.batch_size//chunk, 1)

                batch_data_full = batch[0]

                ## nodal loss
                #### to avoid overcount,
                ## use model.module rather than model in Distribution mode is fine.
                # It works, although I think it is not safe.
                # use model in distribution mode will go wrong, altough it can work in old code version.
                # I suppose it is related to the graph optimization processing in pytorch.
                for chunk_id in range(ng_accu_times):
                    if isinstance(batch_data_full, list):
                        batch_data = torch.cat(
                            [ttt[chunk_id*chunk:(chunk_id+1)*chunk].flatten(1, -1) for ttt in batch_data_full], 1)
                    else:
                        batch_data = batch_data_full[chunk_id *
                                                     chunk:(chunk_id+1)*chunk]
                    ngloss = None
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        if grad_modifier.lambda1 != 0:
                            if ngloss is None:
                                ngloss = 0
                            Nodeloss1 = grad_modifier.getL1loss(model.module if hasattr(
                                model, 'module') else model, batch_data, coef=grad_modifier.coef)/ng_accu_times
                            ngloss += grad_modifier.lambda1 * Nodeloss1
                            Nodeloss1 = Nodeloss1.item()
                        if grad_modifier.lambda2 != 0:
                            if ngloss is None:
                                ngloss = 0
                            Nodeloss2 = grad_modifier.getL2loss(model.module if hasattr(
                                model, 'module') else model, batch_data, coef=grad_modifier.coef)/ng_accu_times
                            ngloss += grad_modifier.lambda2 * Nodeloss2
                            Nodeloss2 = Nodeloss2.item()
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
                loss, abs_loss, iter_info_pool, ltmv_pred, target = run_one_iter(
                    model, batch, criterion, 'train', gpu, data_loader.dataset)

                ## nodal loss
                if (grad_modifier is not None) and (run_gmod) and (grad_modifier.update_mode == 2):
                    if grad_modifier.lambda1 != 0:
                        Nodeloss1 = grad_modifier.getL1loss(
                            model, batch[0], coef=grad_modifier.coef)
                        loss += grad_modifier.lambda1 * Nodeloss1
                        Nodeloss1 = Nodeloss1.item()
                    if grad_modifier.lambda2 != 0:
                        Nodeloss2 = grad_modifier.getL2loss(
                            model, batch[0], coef=grad_modifier.coef)
                        if Nodeloss2 > 0:
                            loss += grad_modifier.lambda2 * Nodeloss2
                        Nodeloss2 = Nodeloss2.item()

            if nan_detect.nan_diagnose_weight(model, loss, loss_scaler):
                continue

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
            if grad_modifier and grad_modifier.path_length_regularize and step % grad_modifier.path_length_regularize == 0:
                mean_path_length = model.mean_path_length.to(device)

                with torch.cuda.amp.autocast(enabled=model.use_amp):
                    path_loss, mean_path_length, path_lengths = grad_modifier.getPathLengthloss(model.module if hasattr(model, 'module') else model,  # <-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch
                                                                                                batch[0], mean_path_length, path_length_mode=grad_modifier.path_length_mode)

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

                if hasattr(model, 'module'):
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
                path_lengths = path_lengths.mean().item()

            rotation_loss = None
            if grad_modifier and grad_modifier.rotation_regularize and step % grad_modifier.rotation_regularize == 0:
                # amp will destroy the train
                #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
                #rotation_loss.backward()
                if grad_modifier.only_eval:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=model.use_amp):
                            rotation_loss = grad_modifier.getRotationDeltaloss(model.module if hasattr(model, 'module') else model,
                                                                               batch[0], ltmv_pred.detach() if len(
                                                                                   batch) == 2 else None, batch[1],
                                                                               rotation_regular_mode=grad_modifier.rotation_regular_mode)
                else:
                    with torch.cuda.amp.autocast(enabled=model.use_amp):
                        rotation_loss = grad_modifier.getRotationDeltaloss(model.module if hasattr(model, 'module') else model,  # <-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch
                                                                           # target,##<-- same reason, should be next stamp rather than last
                                                                           batch[0], ltmv_pred.detach() if len(
                                                                               batch) == 2 else None, batch[1],
                                                                           rotation_regular_mode=grad_modifier.rotation_regular_mode)
                    # notice this y must be x_{t+1}_pred, this works when time_step=2,
                    # when time_step > 2, the provided ltmv_pred is the last pred not the next.
                    if grad_modifier.alpha_stratagy == 'softwarmup50.90':
                        gd_alpha = grad_modifier.gd_alpha * \
                            min(max((np.exp((epoch-50)/40)-1)/(np.exp(1)-1), 0), 1)
                    elif grad_modifier.alpha_stratagy == 'softwarmup00.80':
                        gd_alpha = grad_modifier.gd_alpha * \
                            min(max((np.exp((epoch-0)/80)-1)/(np.exp(1)-1), 0), 1)
                    elif grad_modifier.alpha_stratagy == 'normal':
                        gd_alpha = grad_modifier.gd_alpha
                    else:
                        raise NotImplementedError

                    # default grad_modifier.loss_wall is 0
                    if (rotation_loss > grad_modifier.loss_wall) and gd_alpha > 0:
                        if grad_modifier.loss_target:
                            the_loss = abs(
                                rotation_loss-grad_modifier.loss_target)*gd_alpha
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

            if hasattr(model, 'module') and grad_modifier:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/dist.get_world_size()
                        # <--- pytorch DDP doesn't support high order gradient. This step need!
                        dist.all_reduce(p.grad)

            if model.clip_grad:
                if model.use_amp:
                    assert accumulation_steps == 1
                    loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model.clip_grad)

            if model.directly_esitimate_longterm_error and 'runtime' in model.directly_esitimate_longterm_error:
                assert 'logoffset' in model.directly_esitimate_longterm_error
                normlized_type = normlized_coef_type_bonded
                for key in model.err_record.keys():
                    if hasattr(model, 'module'):
                        dist.barrier()
                        dist.all_reduce(model.err_record[key])
                    model.err_record[key] = model.err_record[key][None]
                c1, c2, c3 = compute_coef(
                    model.err_record, model.directly_esitimate_longterm_error, normlized_type)
                if not hasattr(model, 'clist_buffer'):
                    model.clist_buffer = {'c1': [], 'c2': [], 'c3': []}
                for name, c in zip(['c1', 'c2', 'c3'], [c1.item(), c2.item(), c3.item()]):
                    model.clist_buffer[name].append(c)
                    if len(model.clist_buffer[name]) > 100:
                        model.clist_buffer[name].pop(0)
                        setattr(model, name, np.mean(model.clist_buffer[name]))
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
                    loss, abs_loss, iter_info_pool, ltmv_pred, target = run_one_iter(
                        model, batch, criterion, status, gpu, data_loader.dataset)
                    if optimizer.grad_modifier is not None:
                        if grad_modifier.lambda1 != 0:
                            Nodeloss1 = grad_modifier.getL1loss(
                                model, batch[0], coef=grad_modifier.coef)
                            Nodeloss1 = Nodeloss1.item()
                        if grad_modifier.lambda2 != 0:
                            Nodeloss2 = grad_modifier.getL2loss(
                                model, batch[0], coef=grad_modifier.coef)
                            Nodeloss2 = Nodeloss2.item()

        if logsys.do_iter_log > 0:
            if logsys.do_iter_log == 1:
                iter_info_pool = {}  # disable forward extra information
            iter_info_pool[f'{status}_loss_gpu{gpu}'] = loss.item()
        else:
            iter_info_pool = {}
        if Nodeloss1 > 0:
            iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}'] = Nodeloss1
        if Nodeloss2 > 0:
            iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}'] = Nodeloss2
        if path_loss is not None:
            iter_info_pool[f'{status}_path_loss_gpu{gpu}'] = path_loss
        if path_length is not None:
            iter_info_pool[f'{status}_path_length_gpu{gpu}'] = path_length
        if rotation_loss is not None:
            iter_info_pool[f'{status}_rotation_loss_gpu{gpu}'] = rotation_loss
        total_diff += abs_loss.item()
        #total_num   += len(batch) - 1 #batch
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
                f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
            #print(data_loader.dataset.record_load_tensor.mean().item())
            data_cost = []
            train_cost = []
            rest_cost = []
            inter_b.lwrite(outstring, end="\r")
        #if step>10:break

    if hasattr(model, 'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)

    if model.directly_esitimate_longterm_error and 'during_valid' in model.directly_esitimate_longterm_error and status == 'valid':
        normlized_type = normlized_coef_type2
        if "needbase" in model.directly_esitimate_longterm_error:
            normlized_type = normlized_coef_type3
        elif "vallina" in model.directly_esitimate_longterm_error:
            normlized_type = normlized_coef_type0
        if 'logoffset' in model.directly_esitimate_longterm_error:
            normlized_type = normlized_coef_type_bonded
        if 'per_feature' in model.directly_esitimate_longterm_error:
            for key in model.err_record.keys():
                model.err_record[key] = torch.cat(
                    model.err_record[key]).mean(0)
                if hasattr(model, 'module'):
                    dist.barrier()
                    dist.all_reduce(model.err_record[key])
                model.err_record[key] = model.err_record[key]
            c1, c2, c3 = compute_coef(
                model.err_record, model.directly_esitimate_longterm_error, normlized_type)
        elif 'per_sample' in model.directly_esitimate_longterm_error:
            for key in model.err_record.keys():
                model.err_record[key] = torch.cat(
                    model.err_record[key])  # (B,)
            c1, c2, c3 = compute_coef(
                model.err_record, model.directly_esitimate_longterm_error, normlized_type)
            #print(f"===> before <=== gpu:{device} c1={c1:.4f} c2={c2:.4f} c3={c3:.4f}")
            if hasattr(model, 'module'):
                for x in [c1, c2, c3]:
                    dist.barrier()
                    dist.all_reduce(x)
            #print(f"===> after <=== gpu:{device} c1={c1:.4f} c2={c2:.4f} c3={c3:.4f}")
        else:
            raise NotImplementedError
        model.c1 = c1
        model.c2 = c2
        model.c3 = c3
        model.err_record = {}
        #print(c1,c2,c3)
        #raise

    loss_val = total_diff / total_num
    loss_val = loss_val.item()
    #torch.cuda.empty_cache()
    return loss_val

def run_one_epoch_three2two(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status):
    raise NotImplementedError
    # if status == 'train':
    #     model.train()
    #     logsys.train()
    # elif status == 'valid':
    #     model.eval()
    #     logsys.eval()
    # else:
    #     raise NotImplementedError
    # accumulation_steps = model.accumulation_steps # should be 16 for finetune. but I think its ok.
    # half_model = next(model.parameters()).dtype == torch.float16

    # data_cost  = []
    # train_cost = []
    # rest_cost  = []
    # now = time.time()

    
    # Fethcher   = get_fetcher(status,data_loader)
    # device     = next(model.parameters()).device
    # prefetcher = Fethcher(data_loader,device)
    # #raise
    # batches    = len(data_loader)

    # inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    # gpu        = dist.get_rank() if hasattr(model,'module') else 0

    # if start_step == 0:optimizer.zero_grad()
    # #intervel = batches//100 + 1
    # intervel = batches//logsys.log_trace_times + 1
    # inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    # total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    # nan_count = 0
    # Nodeloss1 = Nodeloss2 = Nodeloss12 = 0
    # path_loss = path_length = rotation_loss = None
    # didunscale = False
    # grad_modifier = optimizer.grad_modifier
    # skip = False
    # count_update = 0
    
    # while inter_b.update_step():
    #     #if inter_b.now>10:break
    #     step = inter_b.now
        
    #     run_gmod = False
    #     if grad_modifier is not None:
    #         control = step if grad_modifier.update_mode==2 else count_update
    #         run_gmod = (control%grad_modifier.ngmod_freq==0)

    #     batch = prefetcher.next()

    #     # In this version(2022-12-22) we will split normal and ngmod processing
    #     # we will do normal train with normal batchsize and learning rate multitimes
    #     # then do ngmod train 
    #     # Notice, one step = once backward not once forward
        
    #     #[print(t[0].shape) for t in batch]
    #     if step < start_step:continue
    #     #batch = data_loader.dataset.do_normlize_data(batch)
        
    #     batch = make_data_regular(batch,half_model)
    #     #if len(batch)==1:batch = batch[0] # for Field -> Field_Dt dataset
    #     data_cost.append(time.time() - now);now = time.time()
    #     assert len(batch) == 3
    #     if status == 'train':
    #         if hasattr(model,'set_step'):model.set_step(step=step,epoch=epoch)
    #         if hasattr(model,'module') and hasattr(model.module,'set_step'):model.module.set_step(step=step,epoch=epoch)
            
    #         # one batch is [(B,P,W,H),(B,P,W,H),(B,P,W,H)]
    #         # the the input should be 
    #         batch = [torch.cat([batch[0],batch[1]]),torch.cat([batch[1],batch[2]])]
    #         with torch.cuda.amp.autocast(enabled=train_config.use_amp):
    #             loss, abs_loss, iter_info_pool,ltmv_pred,target  =run_one_iter(model, batch, criterion, 'train', gpu, data_loader.dataset)
            
    #         loss, nan_count, skip = nan_diagnose_weight(model,loss,nan_count,logsys)
    #         if skip:continue
            
    #         loss /= accumulation_steps
    #         loss_scaler.scale(loss).backward()    

    #         # if train_config.use_amp:
    #         #     loss_scaler.scale(loss).backward()    
    #         # else:
    #         #     loss.backward()
  
    #         rotation_loss = None
    #         if grad_modifier and grad_modifier.rotation_regularize and step%grad_modifier.rotation_regularize==0:
    #             # amp will destroy the train
    #             #rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)
    #             #rotation_loss.backward()
    #             if grad_modifier.only_eval:
    #                 with torch.no_grad(): 
    #                     with torch.cuda.amp.autocast(enabled=train_config.use_amp):
    #                         rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, 
    #                                 batch[0], 
    #                                 ltmv_pred.detach(),
    #                                 target,
    #                                 rotation_regular_mode = grad_modifier.rotation_regular_mode)                     
    #             else:
    #                 with torch.cuda.amp.autocast(enabled=train_config.use_amp):
    #                     rotation_loss= grad_modifier.getRotationDeltaloss(model.module if hasattr(model,'module') else model, #<-- its ok use `model.module`` or `model`, but model.module avoid unknow error of functorch 
    #                             batch[0], ltmv_pred.detach() ,target,rotation_regular_mode = grad_modifier.rotation_regular_mode)                    
    #                 the_loss = rotation_loss*grad_modifier.gd_alpha
    #                 if grad_modifier.use_amp:
    #                     loss_scaler.scale(the_loss).backward()    
    #                 else:
    #                     the_loss.backward()
                        
    #             rotation_loss = rotation_loss.item()


    #         # In order to use multiGPU train, I have to use Loss update scenario, suprisely, it is not worse than split scenario
    #         # if optimizer.grad_modifier is not None:
    #         #     #assert not train_config.use_amp
    #         #     #assert accumulation_steps == 1 
    #         #     if train_config.use_amp and not didunscale:
    #         #         loss_scaler.unscale_(optimizer) # do unscaler here for right gradient modify like clip or norm
    #         #         didunscale = True
    #         #     assert len(batch)==2 # we now only allow one 
    #         #     assert isinstance(batch[0],torch.Tensor)
    #         #     with controlamp(train_config.use_amp)():
    #         #         optimizer.grad_modifier.backward(model, batch[0], batch[1], strict=False)
    #         #         Nodeloss1, Nodeloss12, Nodeloss2 = optimizer.grad_modifier.inference(model, batch[0], batch[1], strict=False)
            

    #         #nan_count, skip = nan_diagnose_grad(model,nan_count,logsys)
    #         # if skip:
    #         #     optimizer.zero_grad()
    #         #     continue
    #         if hasattr(model,'module') and grad_modifier:
    #             for p in model.parameters():
    #                 if p.grad is not None:
    #                     p.grad = p.grad/dist.get_world_size()
    #                     dist.all_reduce(p.grad) #<--- pytorch DDP doesn't support high order gradient. This step need!

    #         if train_config.clip_grad:
    #             if train_config.use_amp:
    #                 assert accumulation_steps == 1
    #                 loss_scaler.unscale_(optimizer)
    #             nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)

    #         if (step+1) % accumulation_steps == 0:
    #             loss_scaler.step(optimizer)
    #             loss_scaler.update()   
    #             # if train_config.use_amp:                  
    #             #     loss_scaler.step(optimizer)
    #             #     loss_scaler.update()   
    #             # else:
    #             #     optimizer.step()
    #             count_update += 1
    #             optimizer.zero_grad()
    #             didunscale = False
    #     else:
    #         with torch.no_grad():
    #             with torch.cuda.amp.autocast(enabled=train_config.use_amp):
    #                 loss, abs_loss, iter_info_pool,ltmv_pred,target =run_one_iter(model, batch, criterion, status, gpu, data_loader.dataset)
    #                 if optimizer.grad_modifier is not None:
    #                     if grad_modifier.lambda1!=0:
    #                         Nodeloss1 = grad_modifier.getL1loss(model, batch[0],coef=grad_modifier.coef)
    #                         Nodeloss1=Nodeloss1.item()
    #                     if grad_modifier.lambda2!=0:
    #                         Nodeloss2 = grad_modifier.getL2loss(model, batch[0],coef=grad_modifier.coef)
    #                         Nodeloss2=Nodeloss2.item()
    #     if logsys.do_iter_log > 0:
    #         if logsys.do_iter_log ==  1:iter_info_pool={} # disable forward extra information
    #         iter_info_pool[f'{status}_loss_gpu{gpu}']       =  loss.item()
    #     else:
    #         iter_info_pool={}
    #     if Nodeloss1  > 0:iter_info_pool[f'{status}_Nodeloss1_gpu{gpu}']  = Nodeloss1
    #     if Nodeloss2  > 0:iter_info_pool[f'{status}_Nodeloss2_gpu{gpu}']  = Nodeloss2
    #     if path_loss is not None:iter_info_pool[f'{status}_path_loss_gpu{gpu}']  = path_loss
    #     if path_length is not None:iter_info_pool[f'{status}_path_length_gpu{gpu}']  = path_length
    #     if rotation_loss is not None:iter_info_pool[f'{status}_rotation_loss_gpu{gpu}']= rotation_loss
    #     total_diff  += abs_loss.item()
    #     #total_num   += len(batch) - 1 #batch 
    #     total_num   += 1 

    #     train_cost.append(time.time() - now);now = time.time()
    #     time_step_now = len(batch)
    #     if (step) % intervel==0:
    #         for key, val in iter_info_pool.items():
    #             logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
    #     if (step) % intervel==0 or step<30:
    #         outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} [RUN Gmod]:{run_gmod}  abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{np.mean(data_cost):.1e} [Train]:{np.mean(train_cost):.1e} ")
    #         #print(data_loader.dataset.record_load_tensor.mean().item())
    #         data_cost  = []
    #         train_cost = []
    #         rest_cost = []
    #         inter_b.lwrite(outstring, end="\r")


    # if hasattr(model,'module') and status == 'valid':
    #     for x in [total_diff, total_num]:
    #         dist.barrier()
    #         dist.reduce(x, 0)

    # loss_val = total_diff/ total_num
    # loss_val = loss_val.item()
    # return loss_val




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
