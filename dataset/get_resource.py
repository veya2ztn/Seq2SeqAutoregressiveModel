
import numpy as np 
import random
import torch
import copy, os
from dataset.WeatherBenchDataset import WeatherBench, FakeWeatherBench
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.tools import get_local_rank
class RandomSelectPatchFetcher:
    def __init__(self,data_loader,device):
        dataset = data_loader.dataset
        assert dataset.use_offline_data  
        self.data  = dataset.dataset_tensor #(B,70,32,64)
        self.batch_size = data_loader.batch_size 
        self.patch_range= dataset.patch_range
        self.img_shape  = dataset.img_shape
        self.around_index = dataset.around_index
        self.center_index = dataset.center_index
        self.length = len(dataset)
        self.time_step = dataset.time_step
        self.device = device
        self.use_time_stamp = dataset.use_time_stamp
        self.use_position_idx= dataset.use_position_idx
        self.timestamp = dataset.timestamp
    def next(self):
        if len(self.img_shape)==2:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2,size=(self.batch_size,)) 
            center_w = np.random.randint(self.img_shape[-1],size=(self.batch_size,))
            location = self.around_index[center_h, center_w] #(B,2,5,5) 
            patch_idx_h = location[:,0]#(B,5,5)
            patch_idx_w = location[:,1]#(B,5,5)
            #location  = self.center_index[:, center_h, center_w].transpose(1,0)#(B,2)
            batch_idx = np.random.randint(self.length,size=(self.batch_size,)).reshape(self.batch_size,1,1) #(B,1,1)
            data = [[self.data[batch_idx+i,:,patch_idx_h,patch_idx_w].permute(0,3,1,2).to(self.device)] for i in range(self.time_step)]
        elif len(self.img_shape)==3:
            center_z    = np.random.randint(self.img_shape[-3] - (self.patch_range[-3]//2)*2,size=(self.batch_size,)) 
            center_h    = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2,size=(self.batch_size,))  
            center_w    = np.random.randint(self.img_shape[-1],size=(self.batch_size,)) 
            location    = self.around_index[center_z, center_h, center_w] #(B,2,5,5,5) 
            patch_idx_z = location[:,0]#(B,5,5,5)
            patch_idx_h = location[:,1]#(B,5,5,5)
            patch_idx_w = location[:,2]#(B,5,5,5)
            #location = self.center_index[:,center_z, center_h, center_w].transpose(1,0)#(B,3)
            batch_idx = np.random.randint(self.length,size=(self.batch_size,)).reshape(self.batch_size,1,1,1) #(B,1,1,1)
            data = [[self.data[batch_idx+i,:,patch_idx_z,patch_idx_h,patch_idx_w].permute(0,4,1,2,3).to(self.device)] for i in range(self.time_step)]
        else:
            raise NotImplementedError
        out = data 
        if self.use_time_stamp:
            out = [out[i]+[torch.Tensor(self.timestamp[batch_idx.flatten()+i]).to(self.device)] for i in range(self.time_step)]
        if self.use_position_idx:
            out = [out[i]+[torch.Tensor(location).to(self.device)] for i in range(self.time_step)]
        if len(out[0])==1:
            out = [t[0] for t in out]
        return out 

class RandomSelectMultiBranchFetcher:
    def __init__(self,data_loader,device):
        dataset = data_loader.dataset
        self.dataset   = dataset
        self.batch_size = data_loader.batch_size 
        self.img_shape  = dataset.img_shape
        self.multibranch_select = multibranch_select = dataset.multibranch_select
        self.time_step  = dataset.time_step
        self.length   = len(dataset) - self.time_step*max(multibranch_select) - 1
        
        self.device   = device
        self.use_time_stamp = dataset.use_time_stamp
        self.timestamp    = dataset.timestamp
    def next(self):
        # we first pin the time_step 
        time_intervel  = int(np.random.choice(self.multibranch_select))
        batch_idx = np.random.randint(self.length, size=(self.batch_size,))# (B,)
        batch = [[torch.from_numpy(np.stack([self.dataset.get_item(start + step*time_intervel) for start in batch_idx])).to(self.device),time_intervel] 
                 for step in range(self.time_step)] 
        return batch 




def seed_worker(worker_id):# Multiprocessing randomnes for multiGPU train #https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_dataloader(args, train_dataset=None, valid_dataset=None, test_dataset=None):
    train_dataloader = valid_dataloader = test_dataloader = None
    if args.Pengine.engine.name == 'naive_distributed':
        if train_dataset is not None:
            train_datasampler = DistributedSampler(train_dataset, shuffle=not args.Train.train_not_shuffle, seed=args.Train.seed) if args.Pengine.engine.distributed else None
            
            g = torch.Generator()
            g.manual_seed(args.Train.seed)
            train_dataloader  = DataLoader(train_dataset, args.Train.batch_size, sampler=train_datasampler, 
                                           num_workers=args.Dataset.dataset.num_workers, pin_memory=True,
                                        drop_last=True,worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True if ((not args.Pengine.engine.distributed) and not args.Train.train_not_shuffle) else False)
        if valid_dataset is not None:
            valid_datasampler = DistributedSampler(valid_dataset,   shuffle=False) if args.Pengine.engine.distributed else None
            valid_dataloader  = DataLoader(valid_dataset  , args.Valid.valid_batch_size, sampler=valid_datasampler,   
                                           num_workers=args.Dataset.dataset.num_workers, pin_memory=True, drop_last=False)
        
    elif args.Pengine.engine.name == 'accelerate':
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        accelerator = args.accelerator
        num_workers = args.Dataset.dataset.num_workers if args.Pengine.engine.data_parallel_dispatch or local_rank == 0 else 0
        if train_dataset is not None:
            train_dataloader = DataLoader(train_dataset, batch_size=args.Train.batch_size,
                                    shuffle=True, num_workers=num_workers, 
                                    pin_memory=True, drop_last=True,)
        if valid_dataset is not None:
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.Train.batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True, drop_last=True,)
        if test_dataset is not None:
            test_dataloader = DataLoader(test_dataset, batch_size=args.Train.batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True, drop_last=True,)
        train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(train_dataloader, valid_dataloader, test_dataloader)
    return   train_dataloader,   valid_dataloader, test_dataloader 

def get_train_and_valid_dataset(args,train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    dataset_type = args.Dataset.dataset.dataset_type
    if get_local_rank(args) and args.Pengine.engine.name == 'accelerate' and not args.Pengine.engine.data_parallel_dispatch:
        dataset_type = 'Fake'+dataset_type
    dataset_type  = eval(dataset_type)
    train_dataset = dataset_type(split="train" if not args.debug else 'test', shared_dataset_tensor_buffer=(train_dataset_tensor, train_record_load), config= args.Dataset.dataset)
    valid_dataset = dataset_type(split="valid" if not args.debug else 'test', shared_dataset_tensor_buffer=(valid_dataset_tensor, valid_record_load), config= args.Dataset.dataset)
    
    train_dataloader,   valid_dataloader, _ = prepare_dataloader(args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=None)

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader

fourcast_default_step ={
    1:20,6: 20, 12: 10, 24:10
}

def get_test_dataset(args,test_dataset_tensor=None,test_record_load=None):
    configs = copy.deepcopy(args.Dataset.dataset)
    configs.time_step = args.Dataset.dataset.time_step if args.Forecast.forecast_future_stamps is None else args.Forecast.forecast_future_stamps
    if configs.time_reverse_flag in ['only_forward', 'random_forward_backward']:configs.time_reverse_flag = 'only_forward'
    dataset_type = eval(args.Dataset.dataset.dataset_type) if isinstance(args.Dataset.dataset.dataset_type,str) else args.Dataset.dataset.dataset_type

    split = args.Forecast.forecast_dataset_split
    test_dataset = dataset_type(split=split, with_idx=True,shared_dataset_tensor_buffer=(test_dataset_tensor,test_record_load),config = config)
    
    # if args.wrapper_model and hasattr(eval(args.wrapper_model),'pred_channel_for_next_stamp'):
    #     test_dataset.pred_channel_for_next_stamp = eval(args.wrapper_model).pred_channel_for_next_stamp
    # if args.multi_branch_order is not None:
    #     test_dataset.multi_branch_order = args.multi_branch_order
    # assert hasattr(test_dataset,'clim_tensor')
    _, _, test_dataloader = prepare_dataloader(args, train_dataset=None, valid_dataset=None, test_dataset=test_dataset)
    
    return   test_dataset,   test_dataloader


def create_memory_templete(args):
    train_dataset_tensor = valid_dataset_tensor = train_record_load = valid_record_load = None
    if args.Dataset.dataset.share_memory:
        assert args.Dataset.dataset.dataset_type
        print("======== loading data as shared memory==========")
        if args.Train.mode not in ['fourcast']:
            # ===============================================================================
            print(f"create training dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.Dataset.dataset.dataset_type
                ).create_offline_dataset_templete(split='train' if not args.debug else 'test',
                                                  create_buffer=True, fully_loaded=False, config=args.Dataset.dataset)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load = train_record_load.share_memory_()
            print(f"done! -> train template shape={train_dataset_tensor.shape}")
            # ===============================================================================
            print(f"create validing dataset template, .....")
            valid_dataset_tensor, valid_record_load = eval(args.Dataset.dataset.dataset_type
                ).create_offline_dataset_templete(split='valid' if not args.debug else 'test',
                                                  create_buffer=True, fully_loaded=False, config=args.Dataset.dataset)
            valid_dataset_tensor = valid_dataset_tensor.share_memory_()
            valid_record_load = valid_record_load.share_memory_()
            print(f"done! -> train template shape={valid_dataset_tensor.shape}")
            # ===============================================================================
        else:
            # ===============================================================================
            print(f"create testing dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.Dataset.dataset.dataset_type
                ).create_offline_dataset_templete(split='test',
                                                  create_buffer=True, fully_loaded=False, config=args.Dataset.dataset)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load = train_record_load.share_memory_()
            print(f"done! -> test template shape={train_dataset_tensor.shape}")
            valid_dataset_tensor = valid_record_load = None
            # ===============================================================================
        print("========      done        ==========")
    return train_dataset_tensor, valid_dataset_tensor, train_record_load, valid_record_load
