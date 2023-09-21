
import torch.nn as nn
from evaluator.utils import compute_accu, compute_rmse
from contextlib import contextmanager
from utils.tools import get_tensor_norm
import torch
import numpy as np
from .forward_step import once_forward, once_forward_normal
from utils.tools import get_local_rank,optional_no_grad
from .forward_step import feature_pick_check

"""
There are two main iteration here
- the normal iteration: do autoregressive N times one the first stamp of given `batch`.
    - use plugin to activate disorder error like loss between X_{t+5}^I  and  X_{t+5}^V
    - use plugin to produce unified rmse/accu regardless the dataengineerning.
- the `run_one_iter_highlevel_fast` is another smart way to compute any order/disorder autoregressive loss as
        shown in 
        X0 X1 X2 X3
        |  |  |  |
        x1 x2 x3 x4
        |  |  |
        y2 y3 y4
        |  |
        z3 z4
  use the `activate_stamps` to assign the computing path.
"""

from .utils import config_of


class RuntimeRMSE_Pligin:
    """
    will convert the 
    """
    one_more_forward = False
    training = False

    def __init__(self, config, normlizer=None):
        assert normlizer is not None
        self.normlizer = normlizer

    def initilize(self):
        pass

    def step(self, model, normlized_predicted_fields: torch.Tensor, normlized_target_fields: torch.Tensor):
        # only deal with the first stamp prediction
        normlized_predicted_fields = normlized_predicted_fields[:, :, 0]
        normlized_target_fields = normlized_target_fields[:, :, 0]
        unnormlized_predicted_fields = self.normlizer.recovery_from_runtime_fields(
            normlized_predicted_fields)
        #(B, P ,W, H)
        unnormlized_target_fields = self.normlizer.recovery_from_runtime_fields(
            normlized_target_fields)
        #(B, P ,W, H)
        with optional_no_grad(not self.training):
            accu = compute_accu(unnormlized_predicted_fields,
                                unnormlized_target_fields).mean().item()
            rmse = compute_rmse(unnormlized_predicted_fields,
                                unnormlized_target_fields).mean().item()
        return {'accu': accu, 'rmse': rmse}


class Consistancy_Plugin:
    one_more_forward = True

    def __init__(self, config, criterion=nn.MSELoss, ):
        self.criterion = criterion()
        self.consistancy_alpha = config.get('consistancy_alpha')
        self.consistancy_cut_grad = config.get('consistancy_cut_grad', False)
        self.vertical_constrain = config.get('vertical_constrain', None)
        self.trainining = not config.get('consistancy_eval', False)

    def initilize(self):
        self.full_fourcast_error_list = []
        self.hidden_fourcast_list = []

    def full_fourcast_forward(self, model, full_fourcast_error_list, ltmv_pred, target, hidden_fourcast_list):
        hidden_fourcast_list_next = []
        extra_loss = 0
        for t in hidden_fourcast_list:
            alpha = self.consistancy_alpha[len(full_fourcast_error_list)]
            if alpha > 0 and t is not None:
                hidden_fourcast = model(t)
                if self.consistancy_cut_grad:
                    hidden_fourcast = hidden_fourcast.detach()
                # can also be criterion(target,hidden_fourcast)
                hidden_error = self.criterion(ltmv_pred, hidden_fourcast)
                hidden_fourcast_list_next.append(hidden_fourcast)
                full_fourcast_error_list.append(hidden_error.item())
                extra_loss += alpha*hidden_error
            else:
                hidden_fourcast_list_next.append(None)
                full_fourcast_error_list.append(0)
        hidden_fourcast_list = hidden_fourcast_list_next + [target]
        #print(full_fourcast_error_list)
        return hidden_fourcast_list, full_fourcast_error_list, extra_loss

    def step(self, model, normlized_predicted_fields: torch.Tensor,
             normlized_target_fields: torch.Tensor):
        # the consistancy_alpha work as follow
        # X0 x1   y2    z3
        #
        #    X1   x2    y3
        #       (100)  (010)
        #         X2    x3
        #              (001)
        #               X3
        loss_for_this_step = {}
        with optional_no_grad(not self.training):
            self.hidden_fourcast_list, self.full_fourcast_error_list, extra_loss2 = self.full_fourcast_forward(model,
                                                                                                               self.full_fourcast_error_list,
                                                                                                               normlized_predicted_fields,
                                                                                                               normlized_target_fields,
                                                                                                               self.hidden_fourcast_list)

        loss_for_this_step['consistancy_loss'] = extra_loss2
        if self.vertical_constrain and len(self.hidden_fourcast_list) >= 2:
            all_hidden_fourcast_list = [
                normlized_predicted_fields]+self.hidden_fourcast_list
            # epsilon_2^I
            first_level_error_tensor = all_hidden_fourcast_list[-1] - \
                all_hidden_fourcast_list[-2]
            for il in range(len(all_hidden_fourcast_list)-2):
                hidden_error_tensor = all_hidden_fourcast_list[-2] - \
                    all_hidden_fourcast_list[il]
                # <epsilon_2^I|epsilon_2^II-epsilon_2^I>
                verticalQ = torch.mean(
                    (hidden_error_tensor*first_level_error_tensor)**2)
                loss_for_this_step['consistancy_vertical_loss'] = verticalQ
        return loss_for_this_step


def run_one_iter(model, batch, criterion, status, sequence_manager, plugins):
    model_config  = config_of(model)
    activate_stamps = getattr(model_config, 'activate_stamps', None)
    if activate_stamps:
        return run_one_iter_highlevel_fast(model, batch, criterion, status, sequence_manager, plugins)
    else:
        return run_one_iter_normal(model, batch, criterion, status, sequence_manager, plugins)
  
def run_one_iter_normal(model, batch, criterion, status, sequence_manager, plugins):
    """
    One iter will forward the model N times.
    N depend on the length of totally sequence, and model_config.pred_len
    
    batch is the entire sequence 
    [
      timestamp_1:{'Field':Field, 'stamp_status':stamp_status]},
      timestamp_2:{'Field':Field, 'stamp_status':stamp_status]}
        ...............................................
      timestamp_n:{'Field':Field, 'stamp_status':stamp_status]}
    ]
    """
    model_config  = config_of(model)
    
    if model_config.history_length > len(batch):
        print.info(f"you want to use history={model_config.history_length}")
        print.info(f"but your input batch(timesteps) only has len(batch)={len(batch)}")
        raise
    
    
    total_forward_times = (len(batch) - model_config.history_length)//model_config.pred_len
    assert (len(batch) - model_config.history_length)%model_config.pred_len==0, f"""
    you must provide correct sequence, based on your assignment, you will use L={model_config.history_length} stamps as inputs and L={model_config.pred_len} as the output.
    However, the data you provide is a sequence L= {len(batch)}
    """
    
    sequence_manager.initial_unnormilized_inputs_field(batch[0:model_config.history_length])
    #plugin  = Consistancy_Plugin()
    _ = [plugin.initilize() for plugin in plugins]

    iter_info_pool = {}
    loss = pred_step = 0
    for i in range(model_config.history_length,len(batch), model_config.pred_len):# i now is the target index
        criterion_now = criterion[pred_step] if isinstance(criterion,(dict,list)) else criterion 
        sequence_manager.push_unnormilized_target_field(batch[i:i+model_config.pred_len])

        
        prediction, normlized_target, sequence_manager = once_forward(model, i, sequence_manager)

        prediction_loss = criterion_now(prediction['field'], normlized_target['field'])
        loss  += prediction_loss 
        iter_info_pool[f'{status}/timestep{i}/prediction_loss'] = prediction_loss.item()
        
        
        if 'structure_loss' in prediction:
            structure_loss = prediction['structure_loss']
            loss  += structure_loss
            iter_info_pool[f'{status}/timestep{i}/structure_loss'] = structure_loss.item()
        
        
        for plugin in plugins:
            plugin_loss = plugin.step(model, prediction['field'], normlized_target)
            for loss_name, loss_val in plugin_loss.items():
                if plugin.training:loss += loss_val
                iter_info_pool[f'{status}/timestep{i}/{plugin}/{loss_name}'] = structure_loss.item()
            
        pred_step += 1
        #if model_config.random_time_step_train and i > np.random.randint(0, total_forward_times):break

                
    
    ####### consistancy will forward last prediction one more times
    plugins = [plugin for plugin in plugins if plugin.one_more_forward]
    if len(plugins)>0:
        sequence_manager.push_unnormilized_target_field(None)
        prediction, normlized_target, sequence_manager = once_forward(model, i, sequence_manager)
        for plugin in plugins:
            plugin_loss = plugin.step(model, prediction['field'], normlized_target)
            for loss_name, loss_val in plugin_loss.items():
                if plugin.training:loss += loss_val
                iter_info_pool[f'{status}/timestep{i}/{plugin}/{loss_name}'] = structure_loss.item()
    
    # loss = loss/(len(batch) - 1)
    
    loss = loss/pred_step
    return loss, iter_info_pool, prediction['field'], normlized_target

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
                if (level_1,level_2,stamp) not in model_config.err_record:model_config.err_record[level_1,level_2,stamp] = []
                model_config.err_record[level_1,level_2,stamp].append(error)

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
                model_config.err_record[level_1,level_2,stamp] = error

            # we would initialize err_reocrd and generate c1,c2,c2 out of the function
        else:
            pass
    else:
        raise # then we directly goes into train mode

    c1,  c2,  c3 = model_config.c1, model_config.c2, model_config.c3
    
    
    
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
    accu = (e1 + e2 + e3)/3
    return loss, accu,iter_info_pool

            
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


def run_one_iter_highlevel_fast(model, batch, criterion, status, sequence_manager, plugins):
    assert model_config.history_length == 1
    assert model_config.pred_len == 1
    assert len(batch)>1
    assert len(batch) <= len(model_config.activate_stamps) + 1
    iter_info_pool={}


    now_level_batch = torch.stack(batch,1) #[(B,P,W,H),(B,P,W,H),...,(B,P,W,H)] -> (B,L,P,W,H)
    # input is a tenosor (B,L,P,W,H)
    # The generated intermediate is recorded as 
    # X0 -> x1 -> y2 -> z3
    # X1 -> x2 -> y3 -> z4
    # X2 -> x3 -> y4
    # X3 -> x4
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
    activate_stamps = model_config.activate_stamps
    if model_config.directly_esitimate_longterm_error and 'during_valid' in model_config.directly_esitimate_longterm_error and status == 'valid':
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
        
        start = [{'field':now_level_batch[:,picked_stamp].flatten(0,1)}]
        sequence_manager.initial_unnormilized_inputs_field(start)
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
            end = [{'field':now_level_batch[:,target_stamp].flatten(0,1)}]
        else:
            end = None
        sequence_manager.push_unnormilized_target_field([end])
        _, _, sequence_manager = once_forward(model, i, sequence_manager)
        now_level_batch        = sequence_manager.input_sequence[-1]['field'].reshape(B,len(picked_stamp),*tshp)  
        #now_level_batch     = model(now_level_batch[:,picked_stamp].flatten(0,1)).reshape(B,len(picked_stamp),*tshp)  
        all_level_batch.append(now_level_batch)
        #all_level_record.append([last_activate_stamp[t]+1 for t in picked_stamp])
        all_level_record.append(activate_stamp)

    ####################################################
    ################ calculate error ###################
    iter_info_pool={}
    loss = 0
    accu = 0
    loss_count = diff_count = len(model_config.activate_error_coef)

    if not model_config.directly_esitimate_longterm_error:
        for (level_1, level_2, stamp, coef,_type) in model_config.activate_error_coef:
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
                accu += coef*error
            
    else:
        loss, accu, iter_info_pool = lets_calculate_the_coef(
            model, model_config.directly_esitimate_longterm_error, status, all_level_batch, all_level_record, iter_info_pool)
        # level_1, level_2, stamp, coef, _type
        # a1 = torch.nn.MSELoss()(all_level_batch[3][:,all_level_record[3].index(3)] , all_level_batch[1][:,all_level_record[1].index(3)])\
        #    /torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[0][:,all_level_record[0].index(2)])
        # a0 = torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[1][:,all_level_record[1].index(2)])\
        #    /torch.nn.MSELoss()(all_level_batch[1][:,all_level_record[1].index(1)] , all_level_batch[0][:,all_level_record[0].index(1)])
        # error = esitimate_longterm_error(a0, a1, model_config.directly_esitimate_longterm_error)
        # iter_info_pool[f"{status}_error_longterm_error_{model_config.directly_esitimate_longterm_error}"] = error.item()
        # loss += error

    return loss, accu, iter_info_pool, None, None

