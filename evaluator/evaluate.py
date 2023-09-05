#########################################
######### fourcast forward step #########
#########################################
def save_and_log_table(_list, logsys, name, column, row=None):
    table= pd.DataFrame(_list.transpose(1,0),index=column, columns=row)
    new_row = [[a]+b for a,b in zip(row,_list.tolist())]
    logsys.add_table(name, new_row , 0, ['fourcast']+column)
    logsys.info(f"===>{name}<===")
    logsys.info(table);
    table.to_csv(os.path.join(logsys.ckpt_root,name))

def get_tensor_value(tensor,snap_index,time = 0):
    regist_batch_id_list, regist_feature_id_list, regist_position = snap_index
    tensor = tensor[0] if isinstance(tensor,list) else tensor
    if isinstance(regist_position,dict):
        regist_position = regist_position[time]
    # regist_batch_id_list is a list for select batch id
    # regist_feature_id_list is a list for select property id
    # regist_position is a position, if is 2D, it should be (#select_points,) (#select_points,)
    if len(tensor.shape) == 5:tensor = tensor.flatten(1,2) 
    output_tensor = []
    for regist_batch_id in regist_batch_id_list:
        one_batch_tensor= []
        for regist_feature_id in regist_feature_id_list:
            if len(regist_position)==2:
                location_tensor= tensor[regist_batch_id][regist_feature_id][regist_position[0],regist_position[1]]
            elif len(regist_position)==3:
                location_tensor= tensor[regist_batch_id,regist_feature_id,regist_position[0],regist_position[1],regist_position[2]]
            else:
                raise NotImplementedError
            one_batch_tensor.append(location_tensor.detach().cpu())
        output_tensor.append(torch.stack(one_batch_tensor))
    return torch.stack(output_tensor)#(B,P,N)

def calculate_next_level_error_once(model, x_t_1, error):
    '''
    calculate m(x_t_1)(x_t_1 - \hat{x_t_1})
    '''
    if  len(x_t_1.shape) == 3:x_t_1=x_t_1[None]
    assert len(x_t_1.shape) == 4
    if  len(error.shape) == 3:error=error[None]
    assert len(error.shape) == 4
    grad       = functorch.jvp(model, (x_t_1,), (error,))[1]  #(B, Xdimension)
    return grad

def calculate_next_level_error(model, x_t_1, error_list):
    '''
    calculate m(x_t_1)(x_t_1 - \hat{x_t_1})
    '''
    assert len(x_t_1.shape) == 4
    grads = vmap(calculate_next_level_error_once, (None,None, 0),randomness='same')(model,x_t_1,error_list)#(N, B, Xdimension)
    return grads

def calculate_next_level_error_batch(model, x_t_1, error_list):
    '''
    calculate m(x_t_1)(x_t_1 - \hat{x_t_1})
    '''
    assert len(x_t_1.shape) == 4
    grads = vmap(calculate_next_level_error_once, (None,0, 0),randomness='same')(model,x_t_1,error_list)#(N, B, Xdimension)
    return grads

def get_tensor_norm(tensor,dim):#<--use mse way
    #return (torch.sum(tensor**2,dim=dim)).sqrt()#(N,B)
    return (torch.mean(tensor**2,dim=dim))#(N,B)

def collect_fourcast_result(fourcastresult,test_dataset,consume=False,force=False):
    offline_out = os.path.join(fourcastresult,"fourcastresult.out")
    if not os.path.exists(offline_out) or force:
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'fourcastresult.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val
                else:
                    if key == 'global_rmse_map':
                        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],tmp['global_rmse_map'])]
                    else:
                        fourcastresult[key] = val # overwrite
        property_names = test_dataset.vnames
        unit_list = test_dataset.unit_list
        if hasattr(test_dataset,"pred_channel_for_next_stamp"):
            offset = 2 if 'CK' in test_dataset.__class__.__name__ else 0
            pred_channel_for_next_stamp = np.array(test_dataset.pred_channel_for_next_stamp) - offset
            property_names = [property_names[t] for t in (pred_channel_for_next_stamp) ] # do not allow padding constant at begining.
            unit_list = unit_list[pred_channel_for_next_stamp]
        
        accu_list = [p['accu'].cpu() for p in fourcastresult.values() if 'accu' in p]
        if not len(accu_list)==0:
            accu_list = torch.stack(accu_list).numpy()   
            total_num = len(accu_list)
            accu_list = accu_list.mean(0)# (fourcast_num,property_num)
            real_times = [(predict_time+1)*test_dataset.time_intervel*test_dataset.time_unit for predict_time in range(len(accu_list))]
            #accu_table = save_and_log_table(accu_list,logsys, prefix+'accu_table', property_names, real_times)    

            ## <============= RMSE ===============>
            rmse_list = torch.stack([p['rmse'].cpu() for p in fourcastresult.values() if 'rmse' in p]).mean(0)# (fourcast_num,property_num)
            #rmse_table= save_and_log_table(rmse_list,logsys, prefix+'rmse_table', property_names, real_times)       

            if not isinstance(unit_list,int):
                unit_list = torch.Tensor(unit_list).to(rmse_list.device)
                #print(unit_list)
                unit_num  = max(unit_list.shape)
                unit_num  = len(property_names)
                unit_list = unit_list.reshape(1,unit_num)
                property_num = len(property_names)
                if property_num > unit_num:
                    assert property_num%unit_num == 0
                    unit_list = torch.repeat_interleave(unit_list,int(property_num//unit_num),dim=1)
            else:
                #logsys.info(f"unit list is int, ")
                unit_list= unit_list
            rmse_unit_list= (rmse_list*unit_list)
            average_metrix = {'accu': accu_list,'rmse':rmse_list,'rmse_unit':rmse_unit_list,'real_times':real_times}
            torch.save(average_metrix,offline_out)
            if consume:
                os.system(f"rm {ROOT}/fourcastresult.gpu_*")
    
    return torch.load(offline_out)

def create_multi_epoch_inference(fourcastresult_path_list, logsys,test_dataset,collect_names=['500hPa_geopotential','850hPa_temperature'],force=False):
    
    origin_ckpt_path = logsys.ckpt_root
    row=[]
    property_names = test_dataset.vnames
    for fourcastresult in fourcastresult_path_list:
        epoch = int(fourcastresult.split("_")[-1])
        assert isinstance(fourcastresult,str)
        prefix = os.path.split(fourcastresult)[-1]
        #logsys.ckpt_root = fourcastresult
        # then it is the fourcastresult path

        result = collect_fourcast_result(fourcastresult, test_dataset,consume=True,force =force)

        if result is not None:
            rmse_unit_list = result['rmse_unit']
            real_times = result['real_times']
            row += [[time_stamp,epoch]+value_list for time_stamp, value_list in zip(real_times,rmse_unit_list.tolist())]
            
            Z500 = rmse_unit_list[real_times.index(120)][property_names.index('500hPa_geopotential')]
            logsys.record('5DZ500', Z500, epoch, epoch_flag='epoch')
            T850 = rmse_unit_list[real_times.index(120)][property_names.index('850hPa_temperature')]
            logsys.record('5DT850', T850, epoch, epoch_flag='epoch')

            Z500 = rmse_unit_list[real_times.index(72)][property_names.index('500hPa_geopotential')]
            logsys.record('3DZ500', Z500, epoch, epoch_flag='epoch')
            T850 = rmse_unit_list[real_times.index(72)][property_names.index('850hPa_temperature')]
            logsys.record('3DT850', T850, epoch, epoch_flag='epoch')

            Z500 = rmse_unit_list[real_times.index(24)][property_names.index('500hPa_geopotential')]
            logsys.record('1DZ500', Z500, epoch, epoch_flag='epoch')
            T850 = rmse_unit_list[real_times.index(24)][property_names.index('850hPa_temperature')]
            logsys.record('1DT850', T850, epoch, epoch_flag='epoch')
    #logsys.add_table(prefix+'_rmse_unit_list', row , 0, ['fourcast']+['epoch'] + property_names)
    logsys.add_table('multi_epoch_fourcast_rmse_unit_list', row , 0, ['fourcast']+['epoch'] + property_names)
    
def get_error_propagation(last_pred, last_target, now_target, now_pred, virtual_function,approx_epsilon_lists, tangent_position='right'):
    #### below is for error esitimation that do not access intermediate level of epsilon
    #### for example, epsilon_{t+6}^{III}. Instead, we use M = J(0.5*X_{t+5}^O + 0.5*X_{t+5}^I ) to calculate error
    #### recurrently. That is  epsilon_{t+6}^{III} = epsilon_{t+6}^{I} + M*epsilon_{t+5}^{II} where epsilon_{t+5}^{II} is
    #### calculated from epsilon_{t+5}^{II} = epsilon_{t+5}^{I} + M*epsilon_{t+4}^{I}
    the_abs_error_measure = None
    the_est_error_measure = None
    epsilon_alevel_v_real = None
    epsilon_blevel_v_real = None
    epsilon_Jacobian_val  = None
    epsilon_Jacobian_valn = None
    epsilon_Jacobian_a    = None
    the_angle_between_two = None
    the_abc_error_measure = None
    if len(approx_epsilon_lists) > 0:
        gradient_value          = last_target # batch[i-1]
        epsilon_alevel_2_real   = now_target - virtual_function(last_target).unsqueeze(0) # ltmv_true - model(last_target) ## epsilon_{t+4}^I
        epsilon_blevel_2_real   = (now_target - now_pred).unsqueeze(0)                                                     ## epsilon_{t+4}^{III}
        normvalue               = get_tensor_norm(approx_epsilon_lists,dim=(2,3,4))
        
        if tangent_position == 'right':
            tangent_x = last_target
        elif tangent_position == 'mid':
            tangent_x = (last_pred + last_target)/2
        elif tangent_position == 'left':
            tangent_x = last_pred
        else:
            raise NotImplementedError

        approx_epsilon_lists    = calculate_next_level_error(virtual_function, tangent_x, approx_epsilon_lists) 
        ## M*epsilon_{t+3}^{I}, M*epsilon_{t+3}^{II}, M*epsilon_{t+3}^{III}, M*epsilon_{t+3}^{IV}
        epsilon_Jacobian_val    = get_tensor_norm(approx_epsilon_lists,dim=(2,3,4))
        epsilon_Jacobian_valn   = epsilon_Jacobian_val/normvalue
        epsilon_Jacobian_val    = epsilon_Jacobian_val[1:]
        epsilon_Jacobian_a      = epsilon_Jacobian_valn[:1] #(1,B)
        epsilon_Jacobian_valn   = epsilon_Jacobian_valn[1:] #(N,B)
        approx_epsilon_lists    = approx_epsilon_lists[1:]  #(N,B)
        epsilon_blevel_2_approx = epsilon_alevel_2_real + approx_epsilon_lists #(N,B,Xdimension) e+m(t)e
        
        the_abs_error_measure   = get_tensor_norm(epsilon_blevel_2_approx - epsilon_blevel_2_real,dim=(2,3,4))#(N,B) # || epsilon_{t+3}^{III} - epsilon_{t+3}^{I} - M epsilon_{t+2}^{II} ||
        the_abc_error_measure   = get_tensor_norm(epsilon_blevel_2_approx,dim=(2,3,4)) - get_tensor_norm(epsilon_blevel_2_real,dim=(2,3,4))#(N,B)
        epsilon_blevel_v_real   = get_tensor_norm(epsilon_blevel_2_real,dim=(3,4))#(N,B,P)
        epsilon_alevel_v_real   = get_tensor_norm(epsilon_alevel_2_real,dim=(3,4))#(N,B,P)
        epsilon_blevel_v_real_norm=epsilon_blevel_v_real.mean(2)#(N,B,P) -> (N,B)
        epsilon_alevel_v_real_norm=epsilon_alevel_v_real.mean(2)#(N,B,P) -> (N,B)
        the_est_error_measure   = (get_tensor_norm(epsilon_blevel_2_real,dim=(2,3,4)) - 
                        get_tensor_norm(epsilon_alevel_2_real,dim=(2,3,4)) - 
                        get_tensor_norm(approx_epsilon_lists,dim=(2,3,4)))#(N,B)
        the_angle_between_two  = torch.einsum('bpwh,nbpwh->nb',epsilon_alevel_2_real.squeeze(0),approx_epsilon_lists
                    )/(torch.sum(epsilon_alevel_2_real**2,dim=(2,3,4)).sqrt()*torch.sum(approx_epsilon_lists**2,dim=(2,3,4)).sqrt())#(N,B)
        approx_epsilon_lists    = torch.cat([epsilon_alevel_2_real, epsilon_blevel_2_real, epsilon_blevel_2_approx]) #(N+2,B,Xdimension)
    else:
        epsilon_alevel_2_real   = (now_target - now_pred).unsqueeze(0)
        epsilon_blevel_2_real   = epsilon_alevel_2_real
        epsilon_blevel_v_real   = get_tensor_norm(epsilon_blevel_2_real,dim=(3,4))
        epsilon_alevel_v_real   = get_tensor_norm(epsilon_alevel_2_real,dim=(3,4))
        approx_epsilon_lists = torch.cat([epsilon_alevel_2_real, epsilon_blevel_2_real])#(2,B,Xdimension)
    
    return (now_target, now_pred,approx_epsilon_lists,
            the_abs_error_measure,
            the_est_error_measure,
            the_abc_error_measure,
            epsilon_alevel_v_real,
            epsilon_blevel_v_real,
            epsilon_Jacobian_val ,
            epsilon_Jacobian_valn,
            epsilon_Jacobian_a,
            the_angle_between_two    )

def recovery_tensor(dataset,start,end,ltmv_pred,target,index=None,model=None):
        if 'Delta' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.combine_base_delta(start[-1][1], start[-1][0]) 
                target    = dataset.combine_base_delta(      end[1],       end[0])  
        elif 'deseasonal' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.addseasonal(start[-1][1], start[-1][0])
                target    = dataset.addseasonal(end[1], end[0])
        elif '68pixelnorm' in dataset.__class__.__name__:
            with torch.no_grad():
                ltmv_pred = dataset.recovery(start[-1])
                target    = dataset.recovery(end)    
        elif ('SPnorm' in dataset.__class__.__name__) or ('Dailynorm' in dataset.__class__.__name__):
            with torch.no_grad():
                ltmv_pred = dataset.recovery(ltmv_pred,index)
                target = dataset.recovery(target, index)
        
        return ltmv_pred,target

def run_one_fourcast_iter(model, batch, idxes, fourcastresult,dataset,**kargs):
    if 'Multibranch' in dataset.__class__.__name__:
        #assert 'MultiBranch' in model.module or model
        return run_one_fourcast_iter_multi_branch(model, batch, idxes, fourcastresult,dataset,**kargs)
    else:
        return run_one_fourcast_iter_single_branch(model, batch, idxes, fourcastresult,dataset,**kargs)

def run_one_fourcast_iter_single_branch(model, batch, idxes, fourcastresult,dataset,
                    save_prediction_first_step=None,save_prediction_final_step=None,
                    snap_index=None,do_error_propagration_monitor=False,**kargs):
    
    smooth_karg = kargs.get('smooth_karg')
    accu_series=[]
    rmse_series=[]
    rmse_maps = []
    hmse_series=[]
    mse_serise= []
    predict_time_series = []
    extra_info = {}
    time_step_1_mode=False
    batch_variance_line_pred = [] 
    batch_variance_line_true = []
    # we will also record the variance for each slot, assume the input is a time series of data
    # [ (B, P, W, H) -> (B, P, W, H) -> .... -> (B, P, W, H) ,(B, P, W, H)]
    # we would record the variance of each batch on the location (W,H)
    error_information = None
    clim = model.clim
    
    start = batch[0:model.history_length] # start must be a list    
    
    snap_line = []
    if (snap_index is not None) and (0 not in [len(t) for t in snap_index]):  
        for i,tensor in enumerate(start):
            # each tensor is like (B, 70, 32, 64) or (B, P, Z, W, H)
            snap_line.append([len(snap_line), get_tensor_value(tensor,snap_index, time=model.history_length),'input'])
    
    # approx_epsilon_lists = []
    # last_pred = last_target = None
    i = model.history_length
    while i<len(batch):#for i in range(model.history_length,len(batch), model.pred_len):# i now is the target index
        
        if do_error_propagration_monitor and i < do_error_propagration_monitor:
            # in this mode, we will concat all batch to accelarate computing. so far, only support batch=[X_{t},X_{t+1},X_{t+2}]
            monitor_batch = torch.stack(batch[:do_error_propagration_monitor],1) #[X_{t},X_{t+1},...,X_{t+2}]-> (B,T,P,W,H)
            ltmv_preds, ltmv_trues,error_information = once_forward_error_evaluation(model,monitor_batch)
            i = do_error_propagration_monitor-1
            start = [ltmv_preds[:,-1]]
            time_list  = range(1,do_error_propagration_monitor)
            ltmv_trues = ltmv_trues.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
            ltmv_preds = ltmv_preds.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
        else:
            end = batch[i:i+model.pred_len]
            end = end[0] if len(end) == 1 else end
            ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward(model,i,start,end,dataset,time_step_1_mode)
            # the index is the timestamp position in dataset
            #print(ltmv_pred.shape)
            ltmv_pred, target = recovery_tensor(
                dataset, start, end, ltmv_pred, target, index=idxes+i*dataset.time_intervel,model=model)
            if hasattr(model,'flag_this_is_shift_model'):
                assert len(start)==2
                ltmv_pred = start[0]
                target    = batch[i-1]
            ltmv_trues = dataset.inv_normlize_data([target])[0]#.detach().cpu() ### use CUDA computing
            ltmv_preds = ltmv_pred#.detach().cpu()
            time_list  = range(i,i+model.pred_len)
            for extra_info_from_model in extra_info_from_model_list:
                for key, val in extra_info_from_model.items():
                    if i not in extra_info:extra_info[i] = {}
                    if key not in extra_info[i]:extra_info[i][key] = []
                    extra_info[i][key].append(val)
        
            if model.pred_len > 1:
                ltmv_trues = ltmv_trues.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
                ltmv_preds = ltmv_preds.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
            else:
                ltmv_trues = ltmv_trues.unsqueeze(0)
                ltmv_preds = ltmv_preds.unsqueeze(0)
        
        i+=model.pred_len         

        #######################################################
        ############### computing processing ##################
        #######################################################
        ### save intermediate cost too much 
        # if save_prediction_first_step is not None and i==model.history_length:save_prediction_first_step[idxes] = ltmv_pred.detach().cpu()
        # if save_prediction_final_step is not None and i==len(batch) - 1:save_prediction_final_step[idxes] = ltmv_pred.detach().cpu()
        for j,(ltmv_true,ltmv_pred) in enumerate(zip(ltmv_trues,ltmv_preds)):
            time = time_list[j]
            ### enter CPU computing
            if len(ltmv_true.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_true = ltmv_true.flatten(1,2) 
            if len(ltmv_pred.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_pred = ltmv_pred.flatten(1,2) 
            if len(clim.shape)!=len(ltmv_pred.shape):
                ltmv_pred = ltmv_pred.squeeze(-1)
                ltmv_true = ltmv_true.squeeze(-1) # temporary use this for timestamp input like [B, P, w,h,T]
            
            if snap_index is not None:
                snap_line.append([time, get_tensor_value(ltmv_pred,snap_index, time=time),'pred'])
                snap_line.append([time, get_tensor_value(ltmv_true,snap_index, time=time),'true'])
            
            predict_time_series.append((j+1)*dataset.time_intervel*dataset.time_unit)
            statistic_dim = tuple(range(2,len(ltmv_true.shape))) # always assume (B,P,Z,W,H)
            batch_variance_line_pred.append(ltmv_pred.std(dim=statistic_dim).detach().cpu())
            batch_variance_line_true.append(ltmv_true.std(dim=statistic_dim).detach().cpu())
            #accu_series.append(compute_accu(ltmv_pred, ltmv_true ).detach().cpu())
            accu_series.append(compute_accu(ltmv_pred - clim.to(ltmv_pred.device), ltmv_true - clim.to(ltmv_pred.device)).detach().cpu())
            rmse_v,rmse_map = compute_rmse(ltmv_pred , ltmv_true, return_map_also=True,**smooth_karg)
            mse = ((ltmv_pred - ltmv_true)**2).mean(dim=(-1,-2))
            mse_serise.append(mse.detach().cpu())
            rmse_series.append(rmse_v.detach().cpu()) #(B,70)
            rmse_maps.append(rmse_map.detach().cpu()) #(70,32,64)
            hmse_value = compute_rmse(ltmv_pred[...,8:24,:], ltmv_true[...,8:24,:],**smooth_karg) if ltmv_pred.shape[-2] == 32 else -torch.ones_like(rmse_v)
            hmse_series.append(hmse_value.detach().cpu())
        #torch.cuda.empty_cache()
    predict_time_series = torch.LongTensor(predict_time_series)#(fourcast_num)
    mse_serise  = torch.stack(mse_serise,1)
    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,property_num)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,property_num)
    hmse_series = torch.stack(hmse_series,1) # (B,fourcast_num,property_num)
    batch_variance_line_pred = torch.stack(batch_variance_line_pred,1) # (B,fourcast_num,property_num)
    batch_variance_line_true = torch.stack(batch_variance_line_true,1) # (B,fourcast_num,property_num)
    
    for idx, mse,accu,rmse,hmse, std_pred,std_true in zip(idxes,mse_serise,accu_series,rmse_series,hmse_series,
                                                batch_variance_line_pred,batch_variance_line_true):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'mse':mse,'accu':accu,"rmse":rmse,'std_pred':std_pred,'std_true':std_true,'snap_line':[],
                                      "hmse":hmse}
    
    if error_information is not None:
        for key, tensor in error_information.items():
            for idx,t in zip(idxes,tensor):
                fourcastresult[idx.item()][key]  = t


    if snap_index is not None:
        for batch_id,select_batch_id in enumerate(snap_index[0]):
            for snap_each_fourcast_time in snap_line:
                # each snap is tensor (b, p, L)
                time_step, tensor, label = snap_each_fourcast_time
                fourcastresult[idxes[select_batch_id].item()]['snap_line'].append([
                    time_step,tensor[batch_id],label
                ])
            #  (p, L) -> (p, L) -> (p, L)
    if "global_rmse_map" not in fourcastresult:
        fourcastresult['global_rmse_map'] = rmse_maps # it is a map list (70,32,64) -> (70, 28,64) ->....
    else:
        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],rmse_maps)]
    return fourcastresult,extra_info

def run_one_fourcast_iter_with_history(model, start, batch, idxes, fourcastresult):
    accu_series=[]
    rmse_series=[]
    out = start
    extra_info = {}
    history_sum_true = history_sum_pred = batch[0].permute(0,2,3,1) if batch[0].shape[1]!=20 or batch[0].shape[1]!=12 else batch[0]
    for i in range(1,len(batch)):
        out   = model(out)
        extra_loss = 0
        if isinstance(out,(list,tuple)):
            extra_loss=out[1]
            for extra_info_from_model in out[2:]:
                for key, val in extra_info_from_model.items():
                    if i not in extra_info:extra_info[i] = {}
                    if key not in extra_info[i]:extra_info[i][key] = []
                    extra_info[i][key].append(val)
            out = out[0]
        ltmv_pred = out.permute(0,2,3,1)# (B, P, W, H ) -> # (B, W, H, P)
        ltmv_true = batch[i].permute(0,2,3,1)# (B, P, W, H ) -> # (B, W, H, P)
        history_sum_pred+=ltmv_pred
        history_sum_true+=ltmv_true
        history_mean_pred=history_sum_pred/(i+1)
        history_mean_true=history_sum_true/(i+1)
        ltmsv_pred = ltmv_pred - history_mean_pred
        ltmsv_true = ltmv_true - history_mean_true
        accu_series.append(compute_accu(ltmsv_pred, ltmsv_true).detach().cpu())
        rmse_series.append(compute_rmse(ltmv_pred , ltmv_true ).detach().cpu())
    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,20)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,20)
    for idx, accu,rmse in zip(idxes,accu_series,rmse_series):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'accu':accu,"rmse":rmse}
    return fourcastresult,extra_info

def compute_multibranch_route(order='do_small_first' ,max_time_step = 150,divide_num=[24, 6, 3, 1]):
    order_table={
        'do_small_first' : [3,2,1,0],
        'do_large_first' : [0,1,2,3]
    }
    if order in order_table:
        order = order_table[order]
        combination_element = {}
        combination_to_target={}
        for target in range(0,max_time_step):
            res = target
            combination_element[target] = []
            for divide in divide_num:
                num = res//divide
                res = res%divide
                combination_element[target].append((num,divide))
            combination_to_target[tuple(combination_element[target])]  = target

        father_to_child = {}
        child_to_father = {}

        for target in range(1,max_time_step-1):
            combination = combination_element[target]
            father_combination = [None,None,None,None]
            reduced = False
            for  i in order:# from 1 -> 6 -> 12 -> 24  or from 24 -> 12 -> 6 -> 1 
                num, element = combination[i]
                if num ==0 or reduced:
                    father_combination[i] = (num,element)
                    continue
                else:
                    father_combination[i] = (num-1,element)
                    reduced = True
            father_combination = tuple(father_combination)
            assert father_combination in combination_to_target, print(father_combination)
            father = combination_to_target[father_combination]
            child_to_father[target] = father
            if father not in father_to_child:father_to_child[father]=[]
            father_to_child[father].append(target)
            father_life_time = {}
            for key, val in father_to_child.items():
                father_life_time[key] = max(val)
        return father_to_child, child_to_father, father_life_time
    else:
        print(f"===> detact we are using {order} mode to control multibranch model")
        if "step" in order: # step_3 step_6 step_1 step_12 step_24
            time_intervel = int(order.split("_")[-1])
            father_to_child  = None
            child_to_father  = dict([(i,i-time_intervel) for i in range(max_time_step)])
            father_life_time = dict([(i-time_intervel,i) for i in range(max_time_step)])
        else:
            raise NotImplementedError

        return father_to_child, child_to_father, father_life_time

def run_one_fourcast_iter_multi_branch(model, batch, idxes, fourcastresult,dataset,
                    save_prediction_first_step=None,save_prediction_final_step=None,
                    snap_index=None,do_error_propagration_monitor=False,order = 'do_large_first',**kargs):
    
    """
    In this branch, the model should be a multibranch model which input model(x, branch_flag) get the prediction for assigned branch.

    The prediction will continuly autoregression to `time_step` prediction target.

    There a many route can be chosed to compute target prediction. For example, one can achieve 7 time_step by 1+6 or 6+1 or 1+1+1+1+1+1.
    Notice the error propagation is different. 

    To achieve the correct prediction, we need provide the compute route for each target. For example, use 1+6 or 6+1 compute 7.

    """
    time_step_1_mode=False
    extra_info = {}
    assert not do_error_propagration_monitor
    assert dataset.time_intervel == 1 # <-- must be 1, temperaly.
    assert model.history_length == 1 # <-- now only support use 1 timestep
    assert model.pred_len == 1
    accu_series              = []
    rmse_series              = []
    rmse_maps                = []
    hmse_series              = []
    mse_serise               = []
    batch_variance_line_pred = []
    batch_variance_line_true = []
    
    # we will also record the variance for each slot, assume the input is a time series of data
    # [ (B, P, W, H) -> (B, P, W, H) -> .... -> (B, P, W, H) ,(B, P, W, H)]
    # we would record the variance of each batch on the location (W,H)
    error_information = None
    clim = model.clim
    
    start = batch[0:model.history_length] # start must be a list    
    snap_line = []
    if (snap_index is not None) and (0 not in [len(t) for t in snap_index]):  
        for i,tensor in enumerate(start):
            snap_line.append([len(snap_line), get_tensor_value(tensor,snap_index, time=model.history_length),'input'])
    
    # We will mantain a buffer that record record "All" computing. 
    # We quote `All` cause, we will delte the buffer if it will not be used anymore. This is realized by mantain a life-line table. 
    prediction_buffer = {0:batch[0]}
    
    father_to_child,  child_to_father, father_life_time_table =  order
    
    
    for i in range(model.history_length,len(batch), model.pred_len):# i now is the target index
        
        #assert len(end) == 2 and end[1]==1 # <-- dataset.time_intervel = 1 and we only return the tensor.
        child_number = i
        father_number = child_to_father[i] 
        if order in ["do_small_first","do_large_first"]:
            assert father_number in prediction_buffer
        else:
            if father_number not in prediction_buffer: # we skip those nonsense data. # notice in this framework we allocate all target together.
                continue
        start = [[prediction_buffer[father_number], child_number - father_number ]]
        end   = [batch[i], None] #<-- set None should be ok
        #print(father_number - child_number)
        ltmv_pred, target, extra_loss, extra_info_from_model_list, start = once_forward_multibranch(model,i,start,end,dataset,time_step_1_mode)
        prediction_buffer[i] = ltmv_pred
        if  father_life_time_table[father_number] <= i: # once the father reach the exit time, we delte it. ## maybe need torch.cuda.empty()
            del prediction_buffer[father_number]
        ltmv_pred, target = recovery_tensor(dataset, start, end, ltmv_pred, target, index=idxes+i*dataset.time_intervel,model=model)
        assert not hasattr(model,'flag_this_is_shift_model')
        ltmv_trues = dataset.inv_normlize_data([target])[0]#.detach().cpu() ### use CUDA computing
        ltmv_preds = ltmv_pred#.detach().cpu()
        time_list  = range(i,i+model.pred_len)
        for extra_info_from_model in extra_info_from_model_list:
            for key, val in extra_info_from_model.items():
                if i not in extra_info:extra_info[i] = {}
                if key not in extra_info[i]:extra_info[i][key] = []
                extra_info[i][key].append(val)
    
        if model.pred_len > 1:
            ltmv_trues = ltmv_trues.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
            ltmv_preds = ltmv_preds.transpose(0,1) #(B,T,P,W,H) -> (T,B,P,W,H)
        else:
            ltmv_trues = ltmv_trues.unsqueeze(0)
            ltmv_preds = ltmv_preds.unsqueeze(0)       

        
        
        
        #######################################################
        ############### computing processing ##################
        #######################################################
        ### save intermediate cost too much 
        # if save_prediction_first_step is not None and i==model.history_length:save_prediction_first_step[idxes] = ltmv_pred.detach().cpu()
        # if save_prediction_final_step is not None and i==len(batch) - 1:save_prediction_final_step[idxes] = ltmv_pred.detach().cpu()
        for j,(ltmv_true,ltmv_pred) in enumerate(zip(ltmv_trues,ltmv_preds)):
            time = time_list[j]
            ### enter CPU computing
            if len(ltmv_true.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_true = ltmv_true.flatten(1,2) 
            if len(ltmv_pred.shape) == 5:#(B,P,Z,W,H) -> (B,P,W,H)
                ltmv_pred = ltmv_pred.flatten(1,2) 
            if len(clim.shape)!=len(ltmv_pred.shape):
                ltmv_pred = ltmv_pred.squeeze(-1)
                ltmv_true = ltmv_true.squeeze(-1) # temporary use this for timestamp input like [B, P, w,h,T]
            
            if snap_index is not None:
                snap_line.append([time, get_tensor_value(ltmv_pred,snap_index, time=time),'pred'])
                snap_line.append([time, get_tensor_value(ltmv_true,snap_index, time=time),'true'])
            
            statistic_dim = tuple(range(2,len(ltmv_true.shape))) # always assume (B,P,Z,W,H)
            batch_variance_line_pred.append(ltmv_pred.std(dim=statistic_dim).detach().cpu())
            batch_variance_line_true.append(ltmv_true.std(dim=statistic_dim).detach().cpu())
            #accu_series.append(compute_accu(ltmv_pred, ltmv_true ).detach().cpu())
            accu_series.append(compute_accu(ltmv_pred - clim.to(ltmv_pred.device), ltmv_true - clim.to(ltmv_pred.device)).detach().cpu())
            rmse_v,rmse_map = compute_rmse(ltmv_pred , ltmv_true, return_map_also=True)
            mse = ((ltmv_pred - ltmv_true)**2).mean(dim=(-1,-2))
            mse_serise.append(mse.detach().cpu())
            rmse_series.append(rmse_v.detach().cpu()) #(B,70)
            rmse_maps.append(rmse_map.detach().cpu()) #(70,32,64)
            hmse_value = compute_rmse(ltmv_pred[...,8:24,:], ltmv_true[...,8:24,:]) if ltmv_pred.shape[-2] == 32 else -torch.ones_like(rmse_v)
            hmse_series.append(hmse_value.detach().cpu())
        #torch.cuda.empty_cache()

    mse_serise  = torch.stack(mse_serise,1)
    accu_series = torch.stack(accu_series,1) # (B,fourcast_num,property_num)
    rmse_series = torch.stack(rmse_series,1) # (B,fourcast_num,property_num)
    hmse_series = torch.stack(hmse_series,1) # (B,fourcast_num,property_num)
    batch_variance_line_pred = torch.stack(batch_variance_line_pred,1) # (B,fourcast_num,property_num)
    batch_variance_line_true = torch.stack(batch_variance_line_true,1) # (B,fourcast_num,property_num)
    
    for idx, mse,accu,rmse,hmse, std_pred,std_true in zip(idxes,mse_serise,accu_series,rmse_series,hmse_series,
                                                batch_variance_line_pred,batch_variance_line_true):
        #if idx in fourcastresult:logsys.info(f"repeat at idx={idx}")
        fourcastresult[idx.item()] = {'mse':mse,'accu':accu,"rmse":rmse,'std_pred':std_pred,'std_true':std_true,'snap_line':[],
                                      "hmse":hmse}
    
    if error_information is not None:
        for key, tensor in error_information.items():
            for idx,t in zip(idxes,tensor):
                fourcastresult[idx.item()][key]  = t


    if snap_index is not None:
        for batch_id,select_batch_id in enumerate(snap_index[0]):
            for snap_each_fourcast_time in snap_line:
                # each snap is tensor (b, p, L)
                time_step, tensor, label = snap_each_fourcast_time
                fourcastresult[idxes[select_batch_id].item()]['snap_line'].append([
                    time_step,tensor[batch_id],label
                ])
            #  (p, L) -> (p, L) -> (p, L)
    if "global_rmse_map" not in fourcastresult:
        fourcastresult['global_rmse_map'] = rmse_maps # it is a map list (70,32,64) -> (70, 28,64) ->....
    else:
        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],rmse_maps)]
    return fourcastresult,extra_info

def fourcast_step(data_loader, model,logsys,random_repeat = 0,snap_index=None,do_error_propagration_monitor=False,order = None,smooth_karg={}):
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
    save_prediction_first_step = None#torch.zeros_like(data_loader.dataset.data)
    save_prediction_final_step = None#torch.zeros_like(data_loader.dataset.data)
    # = 100
    intervel = batches//logsys.log_trace_times + 1
    
    with torch.no_grad():
        inter_b.lwrite("load everything, start_validating......", end="\r")
        while inter_b.update_step():
            #if inter_b.now>10:break
            data_cost += time.time() - now;now = time.time()
            step        = inter_b.now
            idxes,batch = prefetcher.next()
            batch       = make_data_regular(batch,half_model)
            # first sum should be (B, P, W, H )
            the_snap_index_in_iter = None
            if snap_index is not None:
                select_start_timepoints = snap_index[0]
                the_snap_index_in_iter=[[],snap_index[1],snap_index[2]]
                the_snap_index_in_iter[0] = [batch_id for batch_id, idx in enumerate(idxes) if idx in select_start_timepoints]
                if len(the_snap_index_in_iter[0]) == 0: the_snap_index_in_iter=None
            #if the_snap_index_in_iter is None:continue
            with torch.cuda.amp.autocast(enabled=model.use_amp):
                fourcastresult,extra_info = run_one_fourcast_iter(model, batch, idxes, fourcastresult,data_loader.dataset,
                                         save_prediction_first_step=save_prediction_first_step,
                                         save_prediction_final_step=save_prediction_final_step,
                                         snap_index=the_snap_index_in_iter,do_error_propagration_monitor=do_error_propagration_monitor,
                                         order = order,smooth_karg=smooth_karg)
                #print(data_loader.dataset.datatimelist_pool[data_loader.dataset.split][0])
                #property_names = data_loader.dataset.vnames
                #unit_list = data_loader.dataset.unit_list[2:].squeeze()
                #for i, (val,unit,name) in enumerate(zip(fourcastresult[0]["mse"][5],unit_list,property_names)):
                #    print(f"{i:3d} {name:30s} {val*unit:.4f}")
                #print("="*20)
                #unit = unit_list[11].item()
                #name = property_names[11]
                #for i, val in enumerate(fourcastresult[0]["mse"][:,11]):
                #    print(f"{i:3d} {name:30s} {val*unit:.4f}")
                #print("="*20)
                #unit = unit_list[27].item()
                #name = property_names[27]
                #for i, val in enumerate(fourcastresult[0]["mse"][:,27]):
                #    print(f"{i:3d} {name:30s} {val*unit:.4f}")
                #raise
            train_cost += time.time() - now;now = time.time()
            for _ in range(random_repeat):
                raise NotImplementedError
                fourcastresult,extra_info = run_one_fourcast_iter(model, [batch[0]*(1 + torch.randn_like(global_start)*0.05)]+batch[1:], idxes, fourcastresult,data_loader.dataset)
            
            rest_cost += time.time() - now;now = time.time()
            if (step+1) % intervel==0 or step==0:
                for idx, val_pool in extra_info.items():
                    for key, val in val_pool.items():
                        logsys.record(f'test_{key}_each_fourcast_step', np.mean(val), idx, epoch_flag = 'time_step')
                outstring=(f"epoch:fourcast iter:[{step:5d}]/[{len(data_loader)}] GPU:[{gpu}] cost:[Date]:{data_cost/intervel:.1e} [Train]:{train_cost/intervel:.1e} [Rest]:{rest_cost/intervel:.1e}")
                inter_b.lwrite(outstring, end="\r")
            #if inter_b.now >2:break
    if save_prediction_first_step is not None:torch.save(save_prediction_first_step,os.path.join(logsys.ckpt_root,'save_prediction_first_step')) 
    if save_prediction_final_step is not None:torch.save(save_prediction_final_step,os.path.join(logsys.ckpt_root,'save_prediction_final_step')) 
    fourcastresult['snap_index'] = snap_index
    return fourcastresult

def create_fourcast_metric_table(fourcastresult, logsys,test_dataset,collect_names=['500hPa_geopotential','850hPa_temperature'],return_value = None):
    prefix_pool={
        'only_backward':"time_reverse_",
        'only_forward':""
    }
    prefix = prefix_pool[test_dataset.time_reverse_flag]

    if hasattr(test_dataset,'multi_branch_order') and "step" in test_dataset.multi_branch_order:
        test_dataset.time_intervel = int(test_dataset.multi_branch_order.split("_")[-1])

    if isinstance(fourcastresult,str):
        # then it is the fourcastresult path
        ROOT= fourcastresult
        fourcastresult_list = [os.path.join(ROOT,p) for p in os.listdir(fourcastresult) if 'fourcastresult.gpu' in p]
        fourcastresult={}
        for save_path in fourcastresult_list:
            tmp = torch.load(save_path)
            for key,val in tmp.items():
                if key not in fourcastresult:
                    fourcastresult[key] = val
                else:
                    if key == 'global_rmse_map':
                        fourcastresult['global_rmse_map'] = [a+b for a,b in zip(fourcastresult['global_rmse_map'],tmp['global_rmse_map'])]
                    else:
                        fourcastresult[key] = val # overwrite
        

    property_names = test_dataset.vnames
    unit_list = test_dataset.unit_list
    if hasattr(test_dataset,"pred_channel_for_next_stamp"):
        offset = 2 if 'CK' in test_dataset.__class__.__name__ else 0
        pred_channel_for_next_stamp = np.array(test_dataset.pred_channel_for_next_stamp) - offset
        property_names = [property_names[t] for t in (pred_channel_for_next_stamp) ] # do not allow padding constant at begining.
        unit_list = unit_list[pred_channel_for_next_stamp]
    
    # if 'UVTP' in args.wrapper_model:
    #     property_names = [property_names[t] for t in eval(args.wrapper_model).pred_channel_for_next_stamp]
    ## <============= ACCU ===============>
    if 'snap_index' in fourcastresult and fourcastresult['snap_index'] is None:del fourcastresult['snap_index']
    accu_list = torch.stack([p['accu'].cpu() for p in fourcastresult.values() if 'accu' in p]).numpy()    
    total_num = len(accu_list)
    accu_list = accu_list.mean(0)# (fourcast_num,property_num)
    real_times = [(predict_time+1)*test_dataset.time_intervel*test_dataset.time_unit for predict_time in range(len(accu_list))]
    ## <============= RMSE ===============>
    rmse_list = torch.stack([p['rmse'].cpu() for p in fourcastresult.values() if 'rmse' in p]).mean(0)# (fourcast_num,property_num)
    ## <============= HMSE ===============>
    hmse_list = torch.stack([p['hmse'].cpu() for p in fourcastresult.values() if 'hmse' in p]).mean(0)# (fourcast_num,property_num)
    hmse_unit_list = None
    rmse_unit_list = None
    try:
        if not isinstance(unit_list,int):
            unit_list = torch.Tensor(unit_list).to(rmse_list.device)
            #print(unit_list)
            unit_num  = max(unit_list.shape)
            unit_num  = len(property_names)
            unit_list = unit_list.reshape(1,unit_num)
            property_num = len(property_names)
            if property_num > unit_num:
                assert property_num%unit_num == 0
                unit_list = torch.repeat_interleave(unit_list,int(property_num//unit_num),dim=1)
        else:
            logsys.info(f"unit list is int, ")
            unit_list= unit_list
        
        rmse_unit_list= (rmse_list*unit_list)
        
        hmse_unit_list= (hmse_list*unit_list)
        
    except:
        logsys.info(f"get wrong when use unit list, we will fource let [rmse_unit_list] = [rmse_list]")
        traceback.print_exc()

    if return_value is not None:
        assert isinstance(return_value,list) 
        """
        we only check the Z500 
        """
        return rmse_unit_list[real_times.index(120)][property_names.index('500hPa_geopotential')]
    save_and_log_table(accu_list,logsys, prefix+'accu_table', property_names, real_times)    
    save_and_log_table(rmse_list,logsys, prefix+'rmse_table', property_names, real_times)  
    if rmse_unit_list is not None:save_and_log_table(rmse_unit_list,logsys, prefix+'rmse_unit_list', property_names, real_times)     
    if (hmse_list>0).all():save_and_log_table(hmse_list,logsys, prefix+'hmse_table', property_names, real_times)    
    if hmse_unit_list is not None and (hmse_list>0).all():save_and_log_table(hmse_unit_list,logsys, prefix+'hmse_unit_list', property_names, real_times)       
    ## <============= Error_Norm ===============>
    #fourcastresult[idx.item()]['abs_error'] = abs_error
    #fourcastresult[idx.item()]['est_error'] = est_error
    
    ## <============= STD_Location ===============>
    meanofstd = torch.stack([p['std_pred'].cpu() for p in fourcastresult.values() if 'std_pred' in p]).numpy().mean(0)# (B, (fourcast_num,property_num)
    save_and_log_table(meanofstd,logsys, prefix+'meanofstd_table', property_names, real_times)       

    stdofstd = torch.stack([p['std_pred'].cpu() for p in fourcastresult.values() if 'std_pred' in p]).numpy().std(0)# (B, (fourcast_num,property_num)
    save_and_log_table(stdofstd,logsys, prefix+'stdofstd_table', property_names, real_times)      

    
    

    ## <============= Snap_PLot ==================>
    snap_tables = []
    if ('snap_index' in fourcastresult) and (fourcastresult['snap_index'] is not None):
        snap_index = fourcastresult['snap_index']
        select_snap_start_time_point = snap_index[0]
        select_snap_show_property_id = snap_index[1]
        select_snap_show_location    = snap_index[2]
        select_snap_property_name    = [property_names[iidd] for iidd in select_snap_show_property_id]
        for select_time_point in select_snap_start_time_point:
            timestamp = test_dataset.datatimelist_pool['test'][select_time_point]
            if select_time_point in fourcastresult: # in case do not record
                linedata = fourcastresult[select_time_point]['snap_line']
                for predict_time_point, tensor, label in linedata:
                    predict_timestamp = (predict_time_point)*test_dataset.time_intervel*test_dataset.time_unit
                    #  TENSOR --> (P,N) 
                    for propery_name, property_along_location in zip(select_snap_property_name,tensor):
                        for pos_id, value in enumerate(property_along_location):
                            
                            location_x = select_snap_show_location[0][pos_id]
                            location_y = select_snap_show_location[1][pos_id]
                            snap_tables.append([timestamp, label, predict_timestamp,propery_name,location_x,location_y,value])
                    
        logsys.add_table("snap_table", snap_tables , 0, ['start_time',"label","predict_time","propery","pos_x","pos_y","value"])

    if 'global_rmse_map' in fourcastresult:
        global_rmse_map = fourcastresult['global_rmse_map']
        mean_global_rmse_map = [torch.sqrt(t/total_num) for t in global_rmse_map]
        for j,prop_name in enumerate(property_names): 
            if prop_name not in collect_names:continue
            big_step = len(mean_global_rmse_map)
            vmin = min([map_per_time[...,j].min().item() for map_per_time in mean_global_rmse_map])
            vmax = max([map_per_time[...,j].max().item() for map_per_time in mean_global_rmse_map])
            for i,map_per_time in enumerate(mean_global_rmse_map):
                name = f"global_rmse_map_for_{prop_name}"
                s_dir= os.path.join(logsys.ckpt_root,"figures")
                if not os.path.exists(s_dir):os.makedirs(s_dir)
                real_time = real_times[i]
                spath= os.path.join(s_dir,name+f'.{real_time}h.png')
                data = map_per_time[...,j]
                if data.shape[-2] < 32:
                    pad = (32 - data.shape[-2])//2
                    data= torch.nn.functional.pad(data,(0,0,pad,pad),'constant',100)
                assert data.shape[-2] >= 32 
                #images = wandb.Image(mean_global_rmse_map[i][...,j], caption='rmse_map')
                plt.imshow(data.numpy(),vmin=vmin,vmax=vmax,cmap='gray')
                plt.title(f"value range: {vmin:.3f}-{vmax:.3f}")
                plt.xticks([]);plt.yticks([])
                plt.savefig(spath)
                plt.close()
                logsys.wandblog({name:wandb.Image(spath)})
                

    info_pool_list = []
    for predict_time in range(len(accu_list)):
        real_time = real_times[predict_time]
        info_pool={}
        accu_table = accu_list[predict_time]
        rmse_table = rmse_list[predict_time]
        rmse_table_unit = rmse_unit_list[predict_time]

        hmse_table = hmse_list[predict_time]
        hmse_table_unit = hmse_unit_list[predict_time]
        for name,accu,rmse,rmse_unit,hmse,hmse_unit in zip(property_names,accu_table,rmse_table, rmse_table_unit,hmse_table, hmse_table_unit):
            
            info_pool[prefix + f'test_accu_{name}'] = accu.item()
            info_pool[prefix + f'test_rmse_{name}'] = rmse.item()
            info_pool[prefix + f'test_rmse_unit_{name}'] = rmse_unit.item()
            info_pool[prefix + f'test_hmse{name}']    = hmse.item()
            info_pool[prefix + f'test_hmse_unit_{name}'] = hmse_unit.item()
            if real_time in [12, 24, 48, 72, 96, 120]:  
                if name not in collect_names:continue      
                info_pool[prefix + f'{real_time}_hours_test_rmse_unit_{name}'] = rmse_unit.item()
                info_pool[prefix + f'{real_time}_hours_test_rmse_{name}'] = rmse.item()
                info_pool[prefix + f'{real_time}_hours_test_hmse_{name}'] = hmse.item()
                info_pool[prefix + f'{real_time}_hours_test_hmse_unit_{name}'] = hmse_unit.item()
        info_pool['real_time'] = real_time
        for key, val in info_pool.items():
            logsys.record(key,val, predict_time, epoch_flag = 'time_step',extra_epoch_dict = {"real_time":real_time})
        info_pool_list.append(info_pool)

    

    return info_pool_list

def run_fourcast(args, model,logsys,test_dataloader=None,do_table=True,get_value = None):
    import warnings
    warnings.filterwarnings("ignore")
    logsys.info_log_path = os.path.join(logsys.ckpt_root, 'fourcast.info')
    
    if test_dataloader is None:
        test_dataset,  test_dataloader = get_test_dataset(args)

    test_dataset = test_dataloader.dataset
    smooth_karg={'smooth_sigma_1':args.smooth_sigma_1,'smooth_sigma_2':args.smooth_sigma_2,'smooth_times':args.smooth_times}
    order = None
    if args.multi_branch_order is not None:
        divide_num  = list(args.multibranch_select)
        divide_num.sort(reverse=True)
        order = compute_multibranch_route(order=args.multi_branch_order,divide_num=divide_num) 
        print(f"we are using {args.multibranch_select} as branch and do order:{args.multi_branch_order} for divide num {divide_num}")
    #args.force_fourcast=True
    gpu       = dist.get_rank() if hasattr(model,'module') else 0
    fourcastresult_path = os.path.join(logsys.ckpt_root,f"fourcastresult.gpu_{gpu}")
    if not os.path.exists(fourcastresult_path) or  args.force_fourcast:
        if args.force_fourcast and  gpu==0:
            print("re-fourcast, and we will remove old fourcastresult")
            os.system(f"rm {logsys.ckpt_root}/fourcastresult.gpu_*")
        logsys.info(f"use dataset ==> {test_dataset.__class__.__name__}")
        logsys.info("starting fourcast~!")
        with open(os.path.join(logsys.ckpt_root,'weight_path'),'w') as f:f.write(args.pretrain_weight)
        fourcastresult  = fourcast_step(test_dataloader, model,logsys,
                                    random_repeat = args.fourcast_randn_initial,
                                    snap_index=args.snap_index,
                                    do_error_propagration_monitor=args.do_error_propagration_monitor,
                                    order = order,smooth_karg=smooth_karg)
        torch.save(fourcastresult,fourcastresult_path)
        logsys.info(f"save fourcastresult at {fourcastresult_path}")
    else:
        logsys.info(f"load fourcastresult at {fourcastresult_path}")
        fourcastresult = torch.load(fourcastresult_path)

    
    if do_table:
        if not args.distributed:
            create_fourcast_metric_table(fourcastresult, logsys,test_dataset)
        else:
            dist.barrier()
            if dist.get_rank() == 0:
                create_fourcast_metric_table(fourcastresult, logsys,test_dataset)
    if get_value:
        if not args.distributed:
            return create_fourcast_metric_table(fourcastresult, logsys,test_dataset,return_value = ['Z500'])
        else:
            dist.barrier()
            if dist.get_rank() == 0:
                return create_fourcast_metric_table(fourcastresult, logsys,test_dataset,return_value = ['Z500'])
    return -1

def run_fourcast_during_training(args,epoch,logsys,model,test_dataloader):
    if test_dataloader is None:
        test_dataset,  test_dataloader = get_test_dataset(args,test_dataset_tensor=None,test_record_load=None)# should disable at 
    origin_ckpt   =   logsys.ckpt_root
    new_ckpt      =   os.path.join(logsys.ckpt_root,f'result_of_epoch_{epoch}')
    try:# in multi process will conflict
        if new_ckpt and not os.path.exists(new_ckpt):os.makedirs(new_ckpt)
    except:
        pass
    logsys.ckpt_root = new_ckpt
    use_amp = model.use_amp
    #model.use_amp= True ### this should not be changed since some case there has nan
    Z500_now = run_fourcast(args, model,logsys,test_dataloader,do_table=False,get_value=1)
    model.use_amp=use_amp 
    logsys.ckpt_root = origin_ckpt
    return Z500_now,test_dataloader
