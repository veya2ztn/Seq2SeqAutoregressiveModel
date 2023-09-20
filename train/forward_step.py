import torch
from utils.tools import get_tensor_norm
import torch
import numpy as np
#########################################
########## normal forward step ##########
#########################################
"""
    To fit different data input format, like 
    - [list of tensor]
    - [list of [tensor + constant]]
    - .....
Update: 2023.09.06
    We will do unify processing now. All the input should be only for list of dict  
    Notice: there are variout model type like Advection model
"""



def make_data_regular(batch, half_model=False):
    # the input can be
    # [
    #   timestamp_1:[Field,Field_Dt,(physics_part) ],
    #   timestamp_2:[Field,Field_Dt,(physics_part) ],
    #     ...............................................
    #   timestamp_n:[Field,Field_Dt,(physics_part) ]
    # ]
    # or
    # [
    #   timestamp_1:Field,
    #   timestamp_2:Field,
    #     ...............................................
    #   timestamp_n:Field
    # ]
    if not isinstance(batch, (list, tuple)):
        if isinstance(batch, torch.Tensor):
            batch = batch.half() if half_model else batch.float()
            if len(batch.shape) == 4:
                channel_last = batch.shape[1] in [32, 720]  # (B, P, W, H )
                if channel_last:
                    batch = batch.permute(0, 3, 1, 2)
        return batch
    else:
        return [make_data_regular(x, half_model=half_model) for x in batch]


def once_forward_with_timestamp(model, i, start, end, dataset, time_step_1_mode):
    """
    return:
      ltmv_pred    [not normlized pred Field]
      target       [normlized target Field]
      extra_loss
      extra_info_from_model_list
      start        [not normlized Field list]
    """
    
    if not isinstance(end[0], (list, tuple)):end = [end]
    start_timestamp = torch.stack([t[1] for t in start], 1)  # [B,T,4]
    end_timestamp = torch.stack([t[1] for t in end], 1)  # [B,T,4]
    #print([(s[0].shape,s[1].shape) for s in start])
    # start is data list [ [[B,P,h,w],[B,4]] , [[B,P,h,w],[B,4]], [[B,P,h,w],[B,4]], ...]
    normlized_Field_list = dataset.do_normlize_data([[t[0] for t in start]])[0]  # always use normlized input
    normlized_Field = torch.stack(normlized_Field_list, 2)  # (B,P,T,w,h)

    target_list = dataset.do_normlize_data([[t[0] for t in end]])[0]  # always use normlized input
    target = torch.stack(target_list, 2)  # (B,P,T,w,h)

    if 'FED' in model.model_type:
        out = model(normlized_Field, start_timestamp, end_timestamp)
    elif 'Cross' in model.model_type:
        out = model(normlized_Field.squeeze(2), start_timestamp.squeeze(1))
    else:
        out = model(normlized_Field)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    if model.pred_len == 1:
        out = out.squeeze(2)
        target = target.squeeze(2)
    ltmv_pred = dataset.inv_normlize_data([out])[0]
    end_timestamp = end_timestamp.squeeze(1)
    start = start[1:]+[[ltmv_pred, end_timestamp]]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_with_timeconstant(model, i, start, end, dataset, time_step_1_mode):
    assert len(start) == 1
    # start = [[ (B,68,32,64), (B,1,32,64), (B,1,32,64)], [...], [...]]
    # end  = [ (B,68,32,64), (B,1,32,64), (B,1,32,64)]
    normlized_Field = torch.cat(start[0], 1)  # (B,P+2,w,h)
    target = end[0]

    out = model(normlized_Field)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = dataset.inv_normlize_data([out])[0]

    start = start[1:]+[[ltmv_pred]+end[1:]]

    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def feature_pick_check(model):
    unwrapper_model = model
    while hasattr(unwrapper_model, 'module'):
        unwrapper_model = unwrapper_model.module
    model = unwrapper_model
    train_channel_from_this_stamp = getattr(model.config, "train_channel_from_this_stamp", None)
    train_channel_from_next_stamp = getattr(model.config, "train_channel_from_next_stamp", None)
    pred_channel_for_next_stamp   = getattr(model.config, "pred_channel_for_next_stamp", None)
    return train_channel_from_this_stamp, train_channel_from_next_stamp, pred_channel_for_next_stamp

def once_forward_with_Advection(model, i, start, end, dataset, time_step_1_mode):
    # the input can be
    # [
    #   timestamp_1:[Field,Field_Dt,(physics_part) ],
    #   timestamp_2:[Field,Field_Dt,(physics_part) ],
    #     ...............................................
    #   timestamp_n:[Field,Field_Dt,(physics_part) ]
    # ]
    # or
    # [
    #   timestamp_1:Field,
    #   timestamp_2:Field,
    #     ...............................................
    #   timestamp_n:Field
    # ]

    Field = Advection = None

    if isinstance(start[0], (list, tuple)):  # now is [Field, Field_Dt, physics_part]
        Field     = start[-1][0]  # the now Field is the newest timestamp
        Advection = start[-1][-1]
        normlized_Field_list = dataset.do_normlize_data(start)
        normlized_Field_list = [p[0] for p in normlized_Field_list]
        normlized_Field = normlized_Field_list[0] if len(normlized_Field_list) == 1 else torch.stack(normlized_Field_list, 1)
        #(B,P,y,x) for no history case (B,P,H,y,x) for history version
        target = dataset.do_normlize_data([end[0]])[0]  # in standand unit
    elif time_step_1_mode:
        normlized_Field, target = dataset.do_normlize_data([[start, target]])[0]
    else:
        Field = start[-1]
        if hasattr(model, 'calculate_Advection'):Advection = model.calculate_Advection(Field)
        if hasattr(model, 'module') and hasattr(model.module, 'calculate_Advection'):Advection = model.module.calculate_Advection(Field)
        normlized_Field_list = dataset.do_normlize_data([start])[0]  # always use normlized input
        normlized_Field = normlized_Field_list[0] if len(normlized_Field_list) == 1 else torch.stack(normlized_Field_list, 2)
        target = dataset.do_normlize_data([end])[0]  # always use normlized target

    train_channel_from_this_stamp, train_channel_from_next_stamp, pred_channel_for_next_stamp = feature_pick_check(
        model)

    if train_channel_from_this_stamp:
        assert len(normlized_Field.shape) == 4
        normlized_Field = normlized_Field[:, train_channel_from_this_stamp]

    if train_channel_from_next_stamp:
        assert len(normlized_Field.shape) == 4
        normlized_Field = torch.cat(
            [normlized_Field, target[:, train_channel_from_next_stamp]], 1)

    #print(normlized_Field.shape,torch.std_mean(normlized_Field))
    out = model(normlized_Field)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    if Advection is not None:
        normlized_Deltat_F = out
        _, Deltat_F = dataset.inv_normlize_data([[0, normlized_Deltat_F]])[0]
        reduce_Field_coef = torch.Tensor(
            dataset.reduce_Field_coef).to(normlized_Deltat_F.device)
        reduce_Field_coef += getattr(model, "reduce_Field_coef", 0)
        ltmv_pred = Field + Deltat_F - Advection*reduce_Field_coef
    else:
        ltmv_pred = dataset.inv_normlize_data([out])[0]

    if isinstance(start[0], (list, tuple)):
        start = start[1:]+[[ltmv_pred, 0, end[-1]]]
    elif pred_channel_for_next_stamp:
        if target is not None:
            next_tensor = target.clone().type(ltmv_pred.dtype)
            next_tensor[:, pred_channel_for_next_stamp] = ltmv_pred
        else:
            next_tensor = None
        start = start[1:] + [next_tensor]
    # this only let we omit constant pad at level 55 and 69 at test case
    elif (dataset.with_idx and dataset.dataset_flag == '2D70N' and not model.training) or model.skip_constant_2D70N:
        if target is not None:
            next_tensor = target.clone().type(ltmv_pred.dtype)
            picked_property = list(range(0, 14*4-1)) + \
                list(range(14*4, 14*5-1))
            next_tensor[:, picked_property] = ltmv_pred[:, picked_property]
        else:
            next_tensor = ltmv_pred
        start = start[1:] + [next_tensor]
    else:
        start = start[1:] + [ltmv_pred]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))

    if pred_channel_for_next_stamp and target is not None:
        target = target[:, pred_channel_for_next_stamp]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_normal(model, i, sequence_manager, *args):
    """
    forward function only map X_{t+i} to X_{t+i+1}
        model
        i
        start: can either be a single tensor for X_{t} -> X_{t+1} or a tensor chain for  X_{t-n},...,X_{t-1}, X_{t} -> X_{t+1}
        end: the target, can either be a single tensor or a tensor chain for seq2seq model
        time_step_1_mode: force to the single forward mode
    -------------------------------------------------------------------
    # the input can be
    # [
    #   timestamp_1:{'Field':Field, 'stamp_status':stamp_status],
    #   timestamp_2:{'Field':Field, 'stamp_status':stamp_status]
    #     ...............................................
    #   timestamp_n:{'Field':Field, 'stamp_status':stamp_status]
    # ]
    # or
    # [
    #   timestamp_1:{'Field':Field}
    #   timestamp_2:{'Field':Field}
    #     ...............................................
    #   timestamp_n:{'Field':Field}
    # ]
    """

    normlized_fields, normlized_target = sequence_manager.get_inputs_and_target()  # always use normlized target
    
    # if model.training and model.config.input_noise_std and i == 1:
    #     normlized_fields['field'] += torch.randn_like(normlized_fields['field']) * model.config.input_noise_std


    prediction = model(normlized_fields) # --> prediction is also a dict 
    
    # update sequence, update the input sequence and remove the target sequence
    sequence_manager.push_a_normlized_field(prediction['field']) # <-- only the field is needed to update

    return prediction, normlized_target, sequence_manager

def once_forward_multibranch(model, i, start, end, dataset, time_step_1_mode):
    assert len(start) == 1
    assert len(start[0]) == 2  # must be (batch_data, control_flag)
    assert len(end) == 2

    normlized_Field, control_flag = start[0]
    target = end[0]
    #print(normlized_Field.shape,torch.std_mean(normlized_Field))
    
    out = model(normlized_Field, control_flag)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    ltmv_pred = out
    start = [[ltmv_pred, control_flag]]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))

    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start

def once_forward_shift(model, i, start, end, dataset, time_step_1_mode):
    Field = Advection = None
    assert len(start) == 2
    #assert model.shift_feature_index

    model.shift_feature_index = list(range(14*3, 14*4-1))
    normlized_Field = start[0]
    target = start[1].clone()
    target[:, model.shift_feature_index] = end[:, model.shift_feature_index]

    if model.training and model.input_noise_std and i == 1:
        normlized_Field += torch.randn_like(normlized_Field) * \
            model.input_noise_std

    train_channel_from_this_stamp, train_channel_from_next_stamp, pred_channel_for_next_stamp = feature_pick_check(
        model)

    if train_channel_from_this_stamp:
        assert len(normlized_Field.shape) == 4
        normlized_Field = normlized_Field[:, train_channel_from_this_stamp]

    if train_channel_from_next_stamp:
        assert len(normlized_Field.shape) == 4
        normlized_Field = torch.cat(
            [normlized_Field, target[:, train_channel_from_next_stamp]], 1)

    #print(normlized_Field.shape,torch.std_mean(normlized_Field))
    out = model(normlized_Field)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = out

    # notice the target is now be modified as [next_p + now_others]
    assert ltmv_pred.shape[1] == 70 + 13
    end[:, range(14*3, 14*4-1)] = ltmv_pred[:, -13:]
    #####################################
    start = [ltmv_pred[:, :-13], end]
    #<--- this implement is needed, the ltmv_pred used to measure with target should be also the next_p
    new_picked = list(range(14*3)) + list(range(70, 83)) + \
        [14*4-1] + list(range(14*4, 70))
    assert len(new_picked) == 70
    #####################################
    return ltmv_pred[:, new_picked], target, extra_loss, extra_info_from_model_list, start


def once_forward_patch(model, i, start, end, dataset, time_step_1_mode):
    time_stamp = None
    pos = None
    assert len(start) == 1

    start_tensor = start
    if isinstance(start[-1], list):
        assert len(start[-1]) <= 3  # only allow tensor + time_stamp + pos
        tensor, time_stamp, pos = start[-1]
        start_tensor = [tensor]

    Field = start_tensor[-1]
    normlized_Field_list = dataset.do_normlize_data(
        [start_tensor])[0]  # always use normlized input
    normlized_Field = normlized_Field_list[0] if len(
        normlized_Field_list) == 1 else torch.stack(normlized_Field_list, 2)

    if time_stamp is not None or pos is not None:
        target = dataset.do_normlize_data(
            [end[0]])[0]  # always use normlized target
    else:
        target = dataset.do_normlize_data(
            [end])[0]  # always use normlized target

    if model.training and model.input_noise_std and i == 1:
        normlized_Field += torch.randn_like(normlized_Field) * \
            model.input_noise_std

    if (time_stamp is not None) or (pos is not None):
        out = model(normlized_Field, time_stamp, pos)
    else:
        out = model(normlized_Field)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = dataset.inv_normlize_data([out])[0]

    if isinstance(end, (list, tuple)):
        start = start[1:] + [[ltmv_pred, end[1], end[2]]]
    else:
        start = start[1:] + [ltmv_pred]
    #print(ltmv_pred.shape,torch.std_mean(ltmv_pred))
    #print(target.shape,torch.std_mean(target))
    get_center_index_depend_on = model.module.get_center_index_depend_on if hasattr(
        model, 'module') else model.get_center_index_depend_on
    if len(ltmv_pred.shape) > 2:  # (B,P,Z,H,W)
        if ltmv_pred.shape != target.shape:  # (B,P,W,H) -> (B,P,W-4,H) mode
            if len(target.shape) == 5:
                img_shape = target.shape[-3:]
                sld_shape = ltmv_pred.shape[-3:]
                z_idx, h_idx, l_idx = get_center_index_depend_on(sld_shape, img_shape)[
                    0]
                target = target[..., z_idx, h_idx, l_idx]
            elif len(target.shape) == 4:
                img_shape = target.shape[-2:]
                sld_shape = ltmv_pred.shape[-2:]
                h_idx, l_idx = get_center_index_depend_on(
                    sld_shape, img_shape)[0]
                target = target[..., h_idx, l_idx]
            else:
                raise NotImplementedError
        else:
            # (B,P,W,H) -> (B,P,W,H) mode
            pass
    else:  # (B, P)
        #if ltmv_pred.shape[-1] == 1: # (B,P,5,5) -> (B,P) mode
        if len(target.shape) == 4:
            B, P, W, H = target.shape
            target = target[..., W//2, H//2]
        elif len(target.shape) == 5:
            B, P, Z, W, H = target.shape
            target = target[..., Z//2, W//2, H//2]
        else:
            raise NotImplementedError
        # else:
        #     # (B,P,5,5) -> (B,P) mode
        #     target = target
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_patch_N2M(model, i, start, end, dataset, time_step_1_mode):
    time_stamp = None
    pos = None
    assert len(start) == model.history_length
    assert len(end) == model.pred_len
    assert len(start[0]) == 3
    assert len(end[0]) == 3
    # start
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    #         ..........
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]

    # end
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]
    #         ..........
    # [tensor (B,P,W,H), time_stamp (B,4), pos (B,2,W,H)]

    assert isinstance(start[-1], list)

    tensor, start_time_stamp, start_pos = [torch.stack(
        [s[i] for s in start], 1) for i in range(len(start[-1]))]
    target,   end_time_stamp,   end_pos = [torch.stack(
        [s[i] for s in end], 1) for i in range(len(end[-1]))]

    # (B,T,P,W,H) (B,T,4) (B,T,2,W,H)
    out = model(tensor, start_pos, start_time_stamp, end_time_stamp)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]

    ltmv_pred = dataset.inv_normlize_data([out])[0]  # (B,T,P,W,H)

    start = start[len(end):] + [[tensor, time_stamp, pos] for tensor,
                                (_, time_stamp, pos) in zip(ltmv_pred.permute(1, 0, 2, 3, 4), end)]

    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_deltaMode(model, i, start, end, dataset, time_step_1_mode):
    assert len(start) == 1
    base1, delta1 = start[0]
    base2, delta2 = end
    out = model(delta1)

    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    ltmv_pred = out
    target = delta2
    dataset.delta_mean_tensor = dataset.delta_mean_tensor.to(base1.device)
    dataset.delta_std_tensor = dataset.delta_std_tensor.to(base1.device)
    start = start[1:] + [[base1 +
                          (delta1*dataset.delta_std_tensor + dataset.delta_mean_tensor), ltmv_pred]]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_self_relation(model, i, start, end, dataset, time_step_1_mode=None):
    assert len(start) == 1
    input_feature = start[0]
    output_feature = end
    out = model(input_feature)
    extra_loss = 0
    extra_info_from_model_list = []
    if isinstance(out, (list, tuple)):
        extra_loss = out[1]
        extra_info_from_model_list = out[2:]
        out = out[0]
    ltmv_pred = out
    target = output_feature
    start = start[1:] + [None]
    return ltmv_pred, target, extra_loss, extra_info_from_model_list, start


def once_forward_error_evaluation(model, now_level_batch, snap_mode=False):
    target_level_batch = now_level_batch[:, 1:]
    P, W, H = target_level_batch[0, 0].shape
    error_record = []
    real_res_error_record = []
    appx_res_error_record = []
    real_appx_delta_record = []
    appx_res_angle_record = []
    real_res_angle_record = []

    if snap_mode:
        snap_index_w = torch.LongTensor(np.array([18, 17,  1, 15,  6, 27]))
        snap_index_h = torch.LongTensor(np.array([38, 41, 17, 14, 27, 40]))
        snap_index_p = torch.LongTensor(np.array([7, 21, 35, 49, 38]))
        error_record_snap = []
        real_res_error_record_snap = []
    real_res_error = appx_res_error = real_appx_delta = first_level_batch = None
    ltmv_preds = []
    while now_level_batch.shape[1] > 1:
        B, L = now_level_batch.shape[:2]
        tshp = now_level_batch.shape[2:]

        the_whole_tensor = now_level_batch[:, :-1].flatten(0, 1)
        shard_size = 16  # <---- TODO: add this to arguement
        next_level_batch = []
        for shard_index in range(len(the_whole_tensor)//shard_size+1):
            shard_tensor = the_whole_tensor[shard_index *
                                            shard_size:(shard_index+1)*shard_size]
            if len(shard_tensor) == 0:
                break
            next_level_batch.append(model(shard_tensor))
        next_level_batch = torch.cat(next_level_batch)

        next_level_batch = next_level_batch.reshape(B, L-1, *tshp)
        next_level_error_tensor = target_level_batch[:, -(
            L-1):] - next_level_batch
        next_level_error = get_tensor_norm(
            next_level_error_tensor, dim=(3, 4))  # (B,T,P,W,H)->(B,T,P)
        ltmv_preds.append(next_level_batch[:, 0:1])
        error_record.append(next_level_error)

        if snap_mode:
            next_level_error_snap = (
                next_level_error_tensor[:, :, snap_index_p][:, :, :, snap_index_w][:, :, :, :, snap_index_h]**2).detach().cpu()
            error_record_snap.append(next_level_error_snap)
        if first_level_batch is None:
            first_level_batch = next_level_batch
            first_level_error_tensor = next_level_error_tensor
        else:
            real_res_error_tensor = first_level_batch[:, -
                                                      (L-1):] - next_level_batch
            # (B,T,P,W,H)->(B,T,P) # <--record
            real_res_error = get_tensor_norm(real_res_error_tensor, dim=(3, 4))
            if snap_mode:
                real_res_error_snap = (
                    real_res_error_tensor[:, :, snap_index_p][:, :, :, snap_index_w][:, :, :, :, snap_index_h]**2).detach().cpu()
                real_res_error_record_snap.append(real_res_error_snap)
            base_error = first_level_error_tensor[:, -(L-1):]
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
    error_record = torch.cat(error_record,          1).detach().cpu()
    real_res_error_record = torch.cat(real_res_error_record, 1).detach().cpu()
    #real_appx_delta_record= torch.cat(real_appx_delta_record,1).detach().cpu()
    #real_res_angle_record = torch.cat(real_res_angle_record, 1).detach().cpu()
    # appx_res_error_record = torch.cat(appx_res_error_record, 1).detach().cpu()
    # appx_res_angle_record = torch.cat(appx_res_angle_record, 1).detach().cpu()
    if snap_mode:
        error_record_snap = torch.cat(error_record_snap, 1)
        real_res_error_record_snap = torch.cat(real_res_error_record_snap, 1)
    ltmv_preds = torch.cat(ltmv_preds, 1)
    ltmv_trues = target_level_batch
    error_information = {
        "error_record": error_record,
        "real_res_error_record": real_res_error_record,
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

def once_forward(model, i, sequence_manager, *args):
    return once_forward_normal(model, i, sequence_manager, *args)
    # if 'Patch' in dataset.__class__.__name__:
    #     if model.pred_len > 1:
    #         return once_forward_patch_N2M(model, i, start, end, dataset, time_step_1_mode)
    #     else:
    #         return once_forward_patch(model, i, start, end, dataset, time_step_1_mode)
    # elif 'SolarLunaMask' in dataset.__class__.__name__:
    #     return once_forward_with_timeconstant(model, i, start, end, dataset, time_step_1_mode)
    # elif hasattr(dataset, 'use_time_stamp') and dataset.use_time_stamp:
    #     return once_forward_with_timestamp(model, i, start, end, dataset, time_step_1_mode)
    # elif 'Delta' in dataset.__class__.__name__:
    #     return once_forward_deltaMode(model, i, start, end, dataset, time_step_1_mode)
    # elif 'Multibranch' in dataset.__class__.__name__:
    #     return once_forward_multibranch(model, i, start, end, dataset, time_step_1_mode)
    # elif dataset.__class__.__name__ == "WeathBench7066Self":
    #     return once_forward_self_relation(model, i, start, end, dataset, time_step_1_mode)
    # elif hasattr(model, 'flag_this_is_shift_model') or (hasattr(model, 'module') and hasattr(model.module, 'flag_this_is_shift_model')):
    #     return once_forward_shift(model, i, start, end, dataset, time_step_1_mode)

    # else:
    #    return once_forward_normal(model, i, start, end, dataset, time_step_1_mode)
