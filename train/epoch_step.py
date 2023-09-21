from dataset.get_resource import RandomSelectPatchFetcher, RandomSelectMultiBranchFetcher
from mltool.dataaccelerate import DataSimfetcher as Datafetcher
import torch
import time 
import torch.distributed as dist
import numpy as np
from .iter_step import run_one_iter


from criterions.high_order_loss_coef import calculate_coef, normlized_coef_type2, normlized_coef_type3, normlized_coef_type0, normlized_coef_type_bonded

def get_fetcher(status,data_loader):
    return Datafetcher
    if (status =='train' and 
        data_loader.dataset.split=='train' and 
        'Patch' in data_loader.dataset.__class__.__name__):
      return RandomSelectPatchFetcher
    elif (status =='train' and 'Multibranch' in data_loader.dataset.__class__.__name__):
        # notice we should not do valid when use this fetcher
        return RandomSelectMultiBranchFetcher
    else:
        return Datafetcher

def run_one_epoch(status, epoch, start_step,  data_loader, forward_system, sequence_manager, logsys, accelerator, plugins=[]):
    return run_one_epoch_normal(status, epoch, start_step,  data_loader, forward_system, sequence_manager, logsys, accelerator, plugins=plugins)

def apply_model_step(model,*args,**kargs):
    unwrapper_model = model
    while hasattr(unwrapper_model, 'module'):
        unwrapper_model = unwrapper_model.module
    model = unwrapper_model
    if hasattr(model, 'set_step'):model.set_step(*args,**kargs)

class TimeUsagePlugin:
    def __init__(self):
        self.time_cost_list = []
    def record(self, now):
        self.time_cost_list.append(time.time() - now)
        return time.time()
    def get(self):
        average_cost = np.mean(self.time_cost_list)
        self.time_cost_list = []
        return average_cost


class LongTermEstimatePlugin:
    def __init__(self, config):
        self.directly_esitimate_longterm_error = config.get('directly_esitimate_longterm_error')

    @staticmethod
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


    def run_in_epoch(self, model, status):
        if 'runtime' in self.directly_esitimate_longterm_error and status == 'train':
            assert 'logoffset' in self.directly_esitimate_longterm_error
            normlized_type = normlized_coef_type_bonded
            for key in model.err_record.keys():
                if hasattr(model, 'module'):
                    dist.barrier()
                    dist.all_reduce(model.err_record[key])
                model.err_record[key] = model.err_record[key][None]
            c1, c2, c3 = self.compute_coef(model.err_record, self.directly_esitimate_longterm_error, normlized_type)

            if not hasattr(model, 'clist_buffer'):
                model.clist_buffer = {'c1': [], 'c2': [], 'c3': []}
            for name, c in zip(['c1', 'c2', 'c3'], [c1.item(), c2.item(), c3.item()]):
                model.clist_buffer[name].append(c)
                if len(model.clist_buffer[name]) > 100:
                    model.clist_buffer[name].pop(0)
                    setattr(model, name, np.mean(model.clist_buffer[name]))

    def run_end_of_epoch(self, model, status):
        if 'during_valid' in self.directly_esitimate_longterm_error and status == 'valid':
            normlized_type = normlized_coef_type2
            if "needbase" in self.directly_esitimate_longterm_error:
                normlized_type = normlized_coef_type3
            elif "vallina" in self.directly_esitimate_longterm_error:
                normlized_type = normlized_coef_type0
            if 'logoffset' in self.directly_esitimate_longterm_error:
                normlized_type = normlized_coef_type_bonded
            if 'per_feature' in self.directly_esitimate_longterm_error:
                for key in model.err_record.keys():
                    model.err_record[key] = torch.cat(
                        model.err_record[key]).mean(0)
                    if hasattr(model, 'module'):
                        dist.barrier()
                        dist.all_reduce(model.err_record[key])
                    model.err_record[key] = model.err_record[key]
                c1, c2, c3 = self.compute_coef(
                    model.err_record, self.directly_esitimate_longterm_error, normlized_type)
            elif 'per_sample' in self.directly_esitimate_longterm_error:
                for key in model.err_record.keys():
                    model.err_record[key] = torch.cat(model.err_record[key])  # (B,)
                c1, c2, c3 = self.compute_coef(model.err_record, self.directly_esitimate_longterm_error, normlized_type)
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


def run_one_epoch_normal(status, epoch, start_step,  data_loader, forward_system, sequence_manager, logsys, accelerator, plugins=[]):
    model       = forward_system['model']
    criterion   = forward_system.get('criterion',None)
    optimizer   = forward_system.get('optimizer',None)
    loss_scaler = forward_system.get('loss_scaler',None)
    use_amp     = forward_system.get('use_amp',False)
    accumulation_steps     = forward_system.get('accumulation_steps',1)

    if status == 'train':
        model.train()
        logsys.train()
    else:# status == 'valid':
        model.eval()
        logsys.eval()
    if optimizer is not None:optimizer.zero_grad() 


    Fethcher   = get_fetcher(status,data_loader)
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    plugins    = plugins

    data_cost  = TimeUsagePlugin()
    train_cost = TimeUsagePlugin()
    rest_cost  = TimeUsagePlugin()

    batches    = len(data_loader)
    
    intervel = batches//logsys.log_trace_times + 1
    

    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    
    last_record_time = time.time()
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    
    while inter_b.update_step():
        step = inter_b.now
        batch = prefetcher.next()
        #if step < start_step:continue ###, so far will not allowed model checkpoint instep.
        last_record_time = data_cost.record(last_record_time)
        apply_model_step(model,step=step, epoch=epoch, step_total=batches, status=status)
        
        # ====================================================================
        if status == 'train':

            if accelerator is not None:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss, iter_info_pool, _ , _  = run_one_iter(model,batch, criterion, status, sequence_manager, plugins)
                    accelerator.backward(loss)
                    optimizer.step()
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, iter_info_pool, _, _  =run_one_iter(model,batch, criterion, status, sequence_manager, plugins)
                loss_scaler.scale(loss/accumulation_steps).backward()    
                if (step+1) % accumulation_steps == 0:
                    loss_scaler.step(optimizer)
                    loss_scaler.update()   
                    optimizer.zero_grad()
        else:
            with torch.no_grad():
                loss, iter_info_pool, _, _ = run_one_iter(model,batch, criterion, status, sequence_manager, plugins)
        last_record_time = train_cost.record(last_record_time)
        for plugin in plugins:plugin.run_in_epoch(model, status)
        last_record_time = rest_cost.record(last_record_time)

        total_diff  += loss.item()
        total_num   += 1 
        
        
        
        # ====================================================================
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)}  loss:{loss.item():.4f} cost:[Date]:{data_cost.get():.1e} [Train]:{train_cost.get():.1e} ")
            inter_b.lwrite(outstring, end="\r")

    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)
    for plugin in plugins:plugin.run_end_of_epoch(model, status)
    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    return loss_val


