from dataset.utils import RandomSelectPatchFetcher, RandomSelectMultiBranchFetcher
from mltool.dataaccelerate import DataSimfetcher as Datafetcher
import torch
import time 
import torch.distributed as dist
import numpy as np
import torch.nn as nn

from .iter_step import run_one_iter
from .utils import NanDetect
from .forward_step import make_data_regular
from criterions.high_order_loss_coef import calculate_coef, calculate_deltalog_coef, normlized_coef_type2, normlized_coef_type3, normlized_coef_type0, normlized_coef_type_bonded
from utils.tools import get_local_rank, optional_no_grad

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
    return run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler,logsys,status)

def apply_model_step(model,*args,**kargs):
    unwrapper_model = model
    while hasattr(unwrapper_model, 'module'):
        unwrapper_model = unwrapper_model.module
    model = unwrapper_model
    if hasattr(model, 'set_step'):model.set_step(*args,**kargs)

class TimeUsagePlugin:
    def __init__(self, time_cost_list):
        self.time_cost_list = []
    def record(self, now):
        self.time_cost_list.append(time.time() - now)
        return now
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

from sequence2sequence_manager import FieldsSequence
def run_one_epoch_normal(epoch, start_step, model, criterion, data_loader, 
                         optimizer, logsys, status, accelerator, plugins=[]):
    
    if status == 'train':
        model.train()
        logsys.train()
    elif status == 'valid':
        model.eval()
        logsys.eval()
    else:
        raise NotImplementedError
    
    Fethcher   = get_fetcher(status,data_loader)
    device     = next(model.parameters()).device
    prefetcher = Fethcher(data_loader,device)
    plugins    = plugins

    data_cost  = TimeUsagePlugin()
    train_cost = TimeUsagePlugin()
    rest_cost  = TimeUsagePlugin()

    batches    = len(data_loader)
    
    intervel = batches//logsys.log_trace_times + 1
    inter_b.lwrite(f"load everything, start_{status}ing......", end="\r")

    total_diff,total_num  = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    sequence_manager      = FieldsSequence({'batch_size': data_loader})
    last_record_time = time.time()
    inter_b    = logsys.create_progress_bar(batches,unit=' img',unit_scale=data_loader.batch_size)
    while inter_b.update_step():
        #if inter_b.now>10:break
        step = inter_b.now
        batch = prefetcher.next()
        #if step < start_step:continue ###, so far will not allowed model checkpoint instep.
        last_record_time = data_cost.record(last_record_time)
        apply_model_step(model,step=step, epoch=epoch, step_total=batches, status=status)
        if status == 'train':
            with accelerator.accumulate(model): # this cost will not allowed model checkpoint instep. if start_step == 0:optimizer.zero_grad() 
                optimizer.zero_grad()
                loss, abs_loss, iter_info_pool, _ , _  = run_one_iter(model,batch, criterion, status, sequence_manager, plugins)
                accelerator.backward(loss)
                optimizer.step()
        else:
            with torch.no_grad():
                loss, abs_loss, iter_info_pool, _, _ = run_one_iter(model,batch, criterion, status, sequence_manager, plugins)
        last_record_time = train_cost.record(last_record_time)
        for plugin in plugins:plugin.run_in_epoch(model, status)
        last_record_time = rest_cost.record(last_record_time)

        total_diff  += abs_loss.item()
        total_num   += 1 
        
        if (step) % intervel==0:
            for key, val in iter_info_pool.items():
                logsys.record(key, val, epoch*batches + step, epoch_flag='iter')
        if (step) % intervel==0 or step<30:
            outstring=(f"epoch:{epoch:03d} iter:[{step:5d}]/[{len(data_loader)}] [TIME LEN]:{len(batch)} abs_loss:{abs_loss.item():.4f} loss:{loss.item():.4f} cost:[Date]:{data_cost.get():.1e} [Train]:{train_cost.get():.1e} ")
            inter_b.lwrite(outstring, end="\r")

    if hasattr(model,'module') and status == 'valid':
        for x in [total_diff, total_num]:
            dist.barrier()
            dist.reduce(x, 0)
    for plugin in plugins:plugin.run_end_of_epoch(model, status)
    loss_val = total_diff/ total_num
    loss_val = loss_val.item()
    return loss_val


