def full_fourcast_forward(model,criterion,full_fourcast_error_list,ltmv_pred,target,hidden_fourcast_list):
    hidden_fourcast_list_next=[]
    extra_loss=0
    for t in hidden_fourcast_list:
        alpha = model.consistancy_alpha[len(full_fourcast_error_list)]
        if alpha>0 and t is not None:
            hidden_fourcast = model(t )
            if model.consistancy_cut_grad:
                hidden_fourcast=hidden_fourcast.detach()

            hidden_error  = criterion(ltmv_pred,hidden_fourcast) # can also be criterion(target,hidden_fourcast)
            hidden_fourcast_list_next.append(hidden_fourcast)
            full_fourcast_error_list.append(hidden_error.item())
            extra_loss += alpha*hidden_error
        else:
            hidden_fourcast_list_next.append(None)
            full_fourcast_error_list.append(0)
    hidden_fourcast_list = hidden_fourcast_list_next + [target]
    #print(full_fourcast_error_list)
    return hidden_fourcast_list,full_fourcast_error_list,extra_loss


def lets_calculate_the_coef(model, mode, status, all_level_batch, all_level_record, iter_info_pool):
    if 'during_valid' in mode:
        if status == 'valid':
            fixed_activate_error_coef = [[0,1,1],[0,1,2],[0,1,3],[0,2,2],[0,3,3],[1,2,2],[1,3,3]]
            for (level_1, level_2, stamp) in fixed_activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
                if 'per_feature' in mode:
                    error = torch.mean((tensor1-tensor2)**2,dim=(2,3)).detach()
                elif 'per_sample' in mode:
                    error = torch.mean((tensor1-tensor2)**2,dim=(1,2,3)).detach()
                else:
                    error = torch.mean((tensor1-tensor2)**2).detach()
                if (level_1,level_2,stamp) not in model.err_record:model.err_record[level_1,level_2,stamp] = []
                model.err_record[level_1,level_2,stamp].append(error)

            # we would initialize err_reocrd and generate c1,c2,c2 out of the function
        else:
            pass
    elif 'runtime' in mode:
        if status == 'train':
            fixed_activate_error_coef = [[0, 1, 1], [0, 2, 2], [0, 3, 3]]
            for (level_1, level_2, stamp) in fixed_activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
                error = torch.mean((tensor1-tensor2)**2).detach()
                model.err_record[level_1,level_2,stamp] = error

            # we would initialize err_reocrd and generate c1,c2,c2 out of the function
        else:
            pass
    else:
        raise # then we directly goes into train mode

    c1,  c2,  c3 = model.c1, model.c2, model.c3
    
    
    
    if 'per_feature' in mode: # need apply error coef per feature
        error_record = {}
        fixed_activate_error_coef = [[0,1,1],[0,2,2],[0,3,3]]
        for (level_1, level_2, stamp) in fixed_activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
                error_record[level_1,level_2,stamp] = torch.mean((tensor1-tensor2)**2,dim=(0,2,3))
                #in training mode, c1 c2 c3 is also feature-wised
        e1 = error_record[0,1,1]
        e2 = error_record[0,2,2]
        e3 = error_record[0,3,3]
        c1 = c1.mean() if isinstance(c1,torch.Tensor) else c1#(110)
        c2 = c2.mean() if isinstance(c2,torch.Tensor) else c2#(110)
        c3 = c3.mean() if isinstance(c3,torch.Tensor) else c3#(110)
        e1 = e1.mean() if isinstance(e1,torch.Tensor) else e1#(110)
        e2 = e2.mean() if isinstance(e2,torch.Tensor) else e2#(110)
        e3 = e3.mean() if isinstance(e3,torch.Tensor) else e3#(110)
        
    else:
        error_record = {}
        fixed_activate_error_coef = [[0, 1, 1], [0, 2, 2], [0, 3, 3]]
        for (level_1, level_2, stamp) in fixed_activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
                error_record[level_1,level_2,stamp] = torch.mean((tensor1-tensor2)**2)
        if 'deltalog' in mode:
            e1 = torch.log(error_record[0,1,1] + 1)
            e2 = torch.log(error_record[0,2,2] - error_record[0,1,1] + 1)
            e3 = torch.log(error_record[0,3,3] - error_record[0,2,2] + 1)
        elif 'logoffset' in mode:
            offset = 0.01 #<--- this is a hyperparameter
            e1 = torch.log(error_record[0,1,1] + offset)
            e2 = torch.log(error_record[0,2,2] + offset)
            e3 = torch.log(error_record[0,3,3] + offset)
        else: 
            e1 = error_record[0,1,1]
            e2 = error_record[0,2,2]
            e3 = error_record[0,3,3]


    iter_info_pool[f"{status}_c1"] = c1
    iter_info_pool[f"{status}_c2"] = c2
    iter_info_pool[f"{status}_c3"] = c3
    iter_info_pool[f"{status}_e1"] = e1
    iter_info_pool[f"{status}_e2"] = e2
    iter_info_pool[f"{status}_e3"] = e3
    loss = c1*e1 + c2*e2 + c3*e3 
    diff = (e1 + e2 + e3)/3
    return loss, diff,iter_info_pool

from criterions.high_order_loss_coef import calculate_coef,calculate_deltalog_coef,normlized_coef_type2,normlized_coef_type3,normlized_coef_type0,normlized_coef_type_bonded
def run_one_iter_highlevel_fast(model, batch, criterion, status, gpu, dataset):
    assert model.history_length == 1
    assert model.pred_len == 1
    assert len(batch)>1
    assert len(batch) <= len(model.activate_stamps) + 1
    iter_info_pool={}
    
    if model.history_length > len(batch):
        print(f"you want to use history={model.history_length}")
        print(f"but your input batch(timesteps) only has len(batch)={len(batch)}")
        raise
    now_level_batch = torch.stack(batch,1) #[(B,P,W,H),(B,P,W,H),...,(B,P,W,H)] -> (B,L,P,W,H)
    # input is a tenosor (B,L,P,W,H)
    # The generated intermediate is recorded as 
    # X0 x1 y2 z3
    # X1 x2 y3 z4
    # X2 x3 y4
    # X3 x4
    B,L = now_level_batch.shape[:2]
    tshp= now_level_batch.shape[2:]
    all_level_batch = [now_level_batch]
    all_level_record= [list(range(L))] #[0,1,2,3]]
    ####################################################
    # we will do once forward at begin to gain 
    # X0 X1 X2 X3
    # |  |  |  |
    # x1 x2 x3 x4
    # |  |  |
    # y2 y3 y4
    # |  |
    # z3 z4
    ### the problem is we may cut some path by feeding an extra option.
    ### for example, we may achieve a computing graph as
    # X0 X1 X2 X3
    # |  |  |  
    # x1 x2 x3 
    # |   
    # y2 
    # |  
    # z3 
    # so we need a flag 
    ####################################################
    train_channel_from_this_stamp,train_channel_from_next_stamp,pred_channel_for_next_stamp = feature_pick_check(model)
    activate_stamps = model.activate_stamps
    if model.directly_esitimate_longterm_error and 'during_valid' in model.directly_esitimate_longterm_error and status == 'valid':
        activate_stamps = [[1,2,3],[2],[3]]
    for i in range(len(activate_stamps)): # generate L , L-1, L-2
        # the least coding here
        # now_level_batch = model(now_level_batch[:,:(L-i)].flatten(0,1)).reshape(B,(L-i),*tshp)  
        # all_level_batch.append(now_level_batch)
        activate_stamp      = activate_stamps[i]
        last_activate_stamp = all_level_record[-1]
        picked_stamp = []
        for t in activate_stamp:
            picked_stamp.append(last_activate_stamp.index(t-1)) # if t-1 not in last_activate_stamp, raise Error
        start = [now_level_batch[:,picked_stamp].flatten(0,1)]

        if pred_channel_for_next_stamp or train_channel_from_next_stamp:
            if pred_channel_for_next_stamp  : assert t<=L # save key when prediction need last stamp information
            if train_channel_from_next_stamp: assert t< L           
            target_stamp = []
            for t in activate_stamp:
                target_stamp.append(last_activate_stamp.index(t) if t   in last_activate_stamp  else last_activate_stamp.index(t-1))
                # if the target stamp not appear in this batch, use current stamp fill, but we need prohibit this prediction 
                # to do next forward prediction. Thus we limit in t < L 
            # notice when activate pred_channel_for_next_stamp, the unpredicted part should be filled by the part from next stamp 
            # but the loss should be calculate only on the predicted part.
            # In theory, the padded constant will not effect the bask-prapagration. <-- when do average, the padded part will provide a extra length to divide. for example, from 3 element average to 4 element average
            end = now_level_batch[:,target_stamp].flatten(0,1)
        else:
            end = None
        _, _, _, _, start    = once_forward_normal(model,i,start,end,dataset,False)
        now_level_batch      = start[-1].reshape(B,len(picked_stamp),*tshp)  
        #now_level_batch     = model(now_level_batch[:,picked_stamp].flatten(0,1)).reshape(B,len(picked_stamp),*tshp)  
        all_level_batch.append(now_level_batch)
        #all_level_record.append([last_activate_stamp[t]+1 for t in picked_stamp])
        all_level_record.append(activate_stamp)

    ####################################################
    ################ calculate error ###################
    iter_info_pool={}
    loss = 0
    diff = 0
    loss_count = diff_count = len(model.activate_error_coef)

    if not model.directly_esitimate_longterm_error:
        for (level_1, level_2, stamp, coef,_type) in model.activate_error_coef:
            tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
            tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
            if 'quantity' in _type:
                if _type == 'quantity':
                    error   = torch.mean((tensor1-tensor2)**2)
                elif _type == 'quantity_log':
                    error   = ((tensor1-tensor2)**2+1).log().mean()
                elif _type in ['quantity_mean_log','LMSE']:
                    error   = ((tensor1-tensor2)**2).mean().log()
                elif _type in ['quantity_batch_mean_log','PLMSE']:
                    indexes = tuple(np.arange(len(tensor1.shape))[1:])
                    error   = ((tensor1-tensor2)**2).mean(indexes).log().mean()
                elif _type == 'quantity_real_log':
                    error   = ((tensor1-tensor2)**2+1e-2).log().mean()# <---face fatal problem in half precesion due to too small value 
                elif _type == 'quantity_real_log5':
                    error   = ((tensor1-tensor2)**2+1e-5).log().mean()# <---face fatal problem in half precesion due to too small value
                
                elif _type == 'quantity_real_log3':
                    error   = ((tensor1-tensor2)**2+1e-3).log().mean()# <---face fatal problem in half precesion due to too small value 
                    # 1e-2 better than 1e-5. 
                    # May depend on the error unit. For 6 hour task, e around 0.03 so we set 0.01 as offset 
                    # May depend on the error unit. For 1 hour task, e around 0.006 so we may set 0.01 or 0.001 as offset 
                else:raise NotImplementedError
            elif 'alpha' in _type:
                last_tensor1 = all_level_batch[level_1-1][:,all_level_record[level_1-1].index(stamp-1)]
                last_tensor2 = all_level_batch[level_2-1][:,all_level_record[level_2-1].index(stamp-1)]
                if _type == 'alpha':
                    error   = torch.mean(  ((tensor1-tensor2)**2) / ((last_tensor1-last_tensor2)**2+1e-4)  )
                elif _type == 'alpha_log':
                    error   = torch.mean(  ((tensor1-tensor2)**2+1).log() - ((last_tensor1-last_tensor2)**2+1).log() )
                else:raise NotImplementedError
            else:
                raise NotImplementedError
            iter_info_pool[f"{status}_error_{level_1}_{level_2}_{stamp}"] = error.item()
            loss   += coef*error
            if level_1 ==0 and level_2 == stamp:# to be same as normal train 
                diff += coef*error
            
    else:
        loss, diff, iter_info_pool = lets_calculate_the_coef(
            model, model.directly_esitimate_longterm_error, status, all_level_batch, all_level_record, iter_info_pool)
        # level_1, level_2, stamp, coef, _type
        # a1 = torch.nn.MSELoss()(all_level_batch[3][:,all_level_record[3].index(3)] , all_level_batch[1][:,all_level_record[1].index(3)])\
        #    /torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[0][:,all_level_record[0].index(2)])
        # a0 = torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[1][:,all_level_record[1].index(2)])\
        #    /torch.nn.MSELoss()(all_level_batch[1][:,all_level_record[1].index(1)] , all_level_batch[0][:,all_level_record[0].index(1)])
        # error = esitimate_longterm_error(a0, a1, model.directly_esitimate_longterm_error)
        # iter_info_pool[f"{status}_error_longterm_error_{model.directly_esitimate_longterm_error}"] = error.item()
        # loss += error

    return loss, diff, iter_info_pool, None, None
            
def esitimate_longterm_error(a0,a1, n =10):
    """
        # X0 X1 X2
        # |  |  |  
        # x1 x2 x3 
        # |   
        # y2 
        # |  
        # z3 
        a1 = (z3 - x3)/(y2-X2)
        a0 = (y2 - x2)/(x1-X1)
    """
    Bn=1
    for i in range(n):
        Bn = 1 + torch.pow(1-a1,i)*torch.pow(1-a0,1-i)*Bn
    return Bn


def once_forward_error_evaluation(model,now_level_batch,snap_mode = False):
    target_level_batch = now_level_batch[:,1:]
    P,W,H = target_level_batch[0,0].shape
    error_record  = []
    real_res_error_record = []
    appx_res_error_record = []
    real_appx_delta_record  = []
    appx_res_angle_record = []
    real_res_angle_record = []

    if snap_mode:
        snap_index_w = torch.LongTensor(np.array([18, 17,  1, 15,  6, 27]))
        snap_index_h = torch.LongTensor(np.array([38, 41, 17, 14, 27, 40]))
        snap_index_p = torch.LongTensor(np.array([7, 21, 35, 49, 38]))
        error_record_snap = []
        real_res_error_record_snap=[]
    real_res_error = appx_res_error = real_appx_delta = first_level_batch = None
    ltmv_preds =  []
    while now_level_batch.shape[1]>1:
        B,L = now_level_batch.shape[:2]
        tshp= now_level_batch.shape[2:]


        the_whole_tensor = now_level_batch[:,:-1].flatten(0,1)
        shard_size       = 16 #<---- TODO: add this to arguement
        next_level_batch = []
        for shard_index in range(len(the_whole_tensor)//shard_size+1): 
            shard_tensor = the_whole_tensor[shard_index*shard_size:(shard_index+1)*shard_size]
            if len(shard_tensor) == 0:break
            next_level_batch.append(model(shard_tensor))
        next_level_batch = torch.cat(next_level_batch)


        next_level_batch = next_level_batch.reshape(B,L-1,*tshp)
        next_level_error_tensor  = target_level_batch[:,-(L-1):] - next_level_batch 
        next_level_error         = get_tensor_norm(next_level_error_tensor, dim = (3,4)) #(B,T,P,W,H)->(B,T,P)
        ltmv_preds.append(next_level_batch[:, 0:1])
        error_record.append(next_level_error)

        if snap_mode:
            next_level_error_snap = (next_level_error_tensor[:, :, snap_index_p][:,:,:,snap_index_w][:,:,:,:, snap_index_h]**2).detach().cpu()
            error_record_snap.append(next_level_error_snap)
        if first_level_batch is None:
            first_level_batch        = next_level_batch
            first_level_error_tensor = next_level_error_tensor
        else:
            real_res_error_tensor = first_level_batch[:,-(L-1):] - next_level_batch
            real_res_error        = get_tensor_norm(real_res_error_tensor, dim = (3,4)) #(B,T,P,W,H)->(B,T,P) # <--record
            if snap_mode:
                real_res_error_snap = (real_res_error_tensor[:, :, snap_index_p][:, :, :, snap_index_w][:, :, :, :, snap_index_h]**2).detach().cpu()
                real_res_error_record_snap.append(real_res_error_snap)
            base_error            = first_level_error_tensor[:,-(L-1):]
            #real_res_angle        = torch.einsum('btpwh,btpwh->bt',real_res_error_tensor, base_error)/(torch.sum(real_res_error_tensor**2,dim=(2,3,4)).sqrt()*torch.sum(base_error**2,dim=(2,3,4)).sqrt())#->(B,T)
            real_res_error_record.append(real_res_error)
            #real_res_error_alpha  = real_res_error/error_record[-1][:,-(L-1):] #(B,T,P)/(B,T,P)->(B,T,P) # <--can calculate later
            #real_appx_delta       = get_tensor_norm(real_res_error_tensor - appx_res_error_tensor, dim = (3,4)) #(B,T,P,W,H)->(B,T,P) # <--record
            #real_appx_delta_record.append(real_appx_delta)
            #real_res_angle_record.append(real_res_angle)
        
            
            
        # if L>2:
        #     tangent_x             = (target_level_batch[:,-(L-2):] + next_level_batch[:,-(L-2):])/2
        #     appx_res_error_tensor = calculate_next_level_error_batch(model, tangent_x.flatten(0,1), next_level_error_tensor[:,-(L-2):].flatten(0,1)).reshape(B,L-2,*tshp)
        #     base_error = first_level_error_tensor[:,-(L-2):]
        #     appx_res_angle        = torch.einsum('btpwh,btpwh->bt',appx_res_error_tensor, base_error)/(torch.sum(appx_res_error_tensor**2,dim=(2,3,4)).sqrt()*torch.sum(base_error**2,dim=(2,3,4)).sqrt())#->(B,T)
        #     appx_res_error        = get_tensor_norm(appx_res_error_tensor, dim = (3,4)) #(B,T,P,W,H)->(B,T,P) # <--record
        #     appx_res_error_record.append(appx_res_error)
        #     appx_res_angle_record.append(appx_res_angle)
        #     #appx_res_error_alpha  = appx_res_error/error_record[-1][:,-(L-1):]  #(B,T,P)/(B,T,P)->(B,T,P) # <--can calculate later
        now_level_batch = next_level_batch
    error_record          = torch.cat(error_record,          1).detach().cpu()
    real_res_error_record = torch.cat(real_res_error_record, 1).detach().cpu()
    #real_appx_delta_record= torch.cat(real_appx_delta_record,1).detach().cpu()
    #real_res_angle_record = torch.cat(real_res_angle_record, 1).detach().cpu()
    # appx_res_error_record = torch.cat(appx_res_error_record, 1).detach().cpu()
    # appx_res_angle_record = torch.cat(appx_res_angle_record, 1).detach().cpu()
    if snap_mode:
        error_record_snap = torch.cat(error_record_snap, 1)
        real_res_error_record_snap = torch.cat(real_res_error_record_snap, 1)
    ltmv_preds = torch.cat(ltmv_preds,1)
    ltmv_trues = target_level_batch
    error_information={
        "error_record"          :error_record,
        "real_res_error_record" :real_res_error_record,
        #"real_appx_delta_record":real_appx_delta_record,
        #"real_res_angle_record" :real_res_angle_record,
        # "appx_res_error_record" :appx_res_error_record,
        # "appx_res_angle_record" :appx_res_angle_record,
    }
    if snap_mode:
        error_information['error_record_snap'] = error_record_snap
        error_information['real_res_error_record_snap'] = real_res_error_record_snap
        error_information['snap_index_w'] = snap_index_w
        error_information['snap_index_h'] = snap_index_h
        error_information['snap_index_p'] = snap_index_p
    return ltmv_preds, ltmv_trues, error_information

def run_one_iter(model, batch, criterion, status, gpu, dataset):
    if hasattr(model,'activate_stamps') and model.activate_stamps:
        return run_one_iter_highlevel_fast(model, batch, criterion, status, gpu, dataset)
    else:
        return run_one_iter_normal(model, batch, criterion, status, gpu, dataset)

def run_one_iter_normal(model, batch, criterion, status, gpu, dataset):
    iter_info_pool={}
    loss = 0
    diff = 0
    random_run_step = np.random.randint(1,len(batch)) if len(batch)>1 else 0
    time_step_1_mode=False
    if len(batch) == 1 and isinstance(batch[0],(list,tuple)) and len(batch[0])>1:
        batch = batch[0] # (Field, FieldDt)
        time_step_1_mode=True
    if model.history_length > len(batch):
        print(f"you want to use history={model.history_length}")
        print(f"but your input batch(timesteps) only has len(batch)={len(batch)}")
        raise
    pred_step = 0
    start = batch[0:model.history_length] # start must be a list
    full_fourcast_error_list = []
    hidden_fourcast_list = [] 
    # length depend on pred_len time_step=2 --> 1+1
    #                time_step=3 --> 1+2+2
    #                time_step=4 --> 1+2+3
    # use_consistancy_alpha to control activate amplitude
    
    
    for i in range(model.history_length,len(batch), model.pred_len):# i now is the target index
        end = batch[i:i+model.pred_len]
        end = end[0] if len(end) == 1 else end
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,end,dataset,time_step_1_mode)
        
            
        if extra_loss !=0:
            iter_info_pool[f'{status}_extra_loss_gpu{gpu}_timestep{i}'] = extra_loss.item()
        for extra_info_from_model in extra_info_from_model_list:
            for name, value in extra_info_from_model.items():
                iter_info_pool[f'{status}_on_{status}_{name}_timestep{i}'] = value
        
        ltmv_pred = dataset.do_normlize_data([ltmv_pred])[0]

        if 'Delta' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)+ extra_loss
            with torch.no_grad():
                normlized_field_predict = dataset.combine_base_delta(start[-1][0], start[-1][1]) 
                normlized_field_real    = dataset.combine_base_delta(      end[0],       end[1])  
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
            
        elif 'deseasonal' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)+ extra_loss
            with torch.no_grad():
                normlized_field_predict = dataset.addseasonal(start[-1][0], start[-1][1])
                normlized_field_real    = dataset.addseasonal(end[0], end[1])
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
        elif '68pixelnorm' in dataset.__class__.__name__:
            loss  += criterion(ltmv_pred,target)+ extra_loss
            with torch.no_grad():
                normlized_field_predict = dataset.recovery(start[-1])
                normlized_field_real    = dataset.recovery(end)
                abs_loss = criterion(normlized_field_predict,normlized_field_real)
        else:
            normlized_field_predict = ltmv_pred
            normlized_field_real = target
            abs_loss = criterion[pred_step](ltmv_pred,target) if isinstance(criterion,(dict,list)) else criterion(ltmv_pred,target)
            loss += abs_loss + extra_loss
        
        diff += abs_loss
        pred_step+=1
        
        if hasattr(model,"consistancy_alpha") and model.consistancy_alpha and loss < model.consistancy_activate_wall and ((not model.consistancy_eval) or (not model.training)): 
            # the consistancy_alpha work as follow
            # X0 x1   y2    z3
            #    
            #    X1   x2    y3
            #       (100)  (010)
            #         X2    x3
            #              (001)
            #               X3
            hidden_fourcast_list,full_fourcast_error_list,extra_loss2 = full_fourcast_forward(model,criterion,full_fourcast_error_list,ltmv_pred,target,hidden_fourcast_list)
            if hasattr(model,"vertical_constrain") and model.vertical_constrain and len(hidden_fourcast_list)>=2:
                all_hidden_fourcast_list = [ltmv_pred]+hidden_fourcast_list
                first_level_error_tensor = all_hidden_fourcast_list[-1] - all_hidden_fourcast_list[-2] #epsilon_2^I
                for il in range(len(all_hidden_fourcast_list)-2):
                    hidden_error_tensor = all_hidden_fourcast_list[-2] - all_hidden_fourcast_list[il]
                    verticalQ = torch.mean((hidden_error_tensor*first_level_error_tensor)**2) # <epsilon_2^I|epsilon_2^II-epsilon_2^I>
                    iter_info_pool[f'{status}_vertical_error_{i}_{il}_gpu{gpu}'] =  verticalQ.item()
                    extra_loss2+= model.vertical_constrain*verticalQ # we only
            if not model.consistancy_eval:loss+= extra_loss2
            
                
        iter_info_pool[f'{status}_abs_loss_gpu{gpu}_timestep{i}'] =  abs_loss.item()
        if status != "train":
            iter_info_pool[f'{status}_accu_gpu{gpu}_timestep{i}']     =  compute_accu(normlized_field_predict,normlized_field_real).mean().item()
            iter_info_pool[f'{status}_rmse_gpu{gpu}_timestep{i}']     =  compute_rmse(normlized_field_predict,normlized_field_real).mean().item()
        if model.random_time_step_train and i >= random_run_step:
            break
    if hasattr(model,"consistancy_alpha") and model.consistancy_alpha and loss < model.consistancy_activate_wall and ((not model.consistancy_eval) or (not model.training)): 
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,None,dataset,time_step_1_mode) 
        ltmv_pred = dataset.do_normlize_data([ltmv_pred])[0]
        hidden_fourcast_list,full_fourcast_error_list,extra_loss2 = full_fourcast_forward(model,criterion,full_fourcast_error_list,ltmv_pred,None,hidden_fourcast_list)
        if not model.consistancy_eval:loss+= extra_loss2
        for iii, val in enumerate(full_fourcast_error_list):
            if val >0: iter_info_pool[f'{status}_full_fourcast_error_{iii}_gpu{gpu}'] =  val
    # loss = loss/(len(batch) - 1)
    # diff = diff/(len(batch) - 1)
    loss = loss/pred_step
    diff = diff/pred_step
    return loss, diff, iter_info_pool, ltmv_pred, target
