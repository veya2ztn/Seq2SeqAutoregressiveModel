
import numpy as np 
import random
import torch
import copy
from dataset.WeatherBenchDataset import WeatherBench

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

def get_train_and_valid_dataset(args,train_dataset_tensor=None,train_record_load=None,valid_dataset_tensor=None,valid_record_load=None):
    dataset_type   = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    
    train_dataset  = dataset_type(split="train" if not args.debug else 'test',dataset_tensor=train_dataset_tensor,
                                  record_load_tensor=train_record_load,**args.dataset_kargs)
    val_dataset   = dataset_type(split="valid" if not args.debug else 'test',dataset_tensor=valid_dataset_tensor,
                                  record_load_tensor=valid_record_load,**args.dataset_kargs)
    
    train_datasampler = DistributedSampler(train_dataset, shuffle=args.do_train_shuffle, seed=args.Train.seed) if args.distributed else None
    val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None
    g = torch.Generator()
    g.manual_seed(args.Train.seed)
    train_dataloader  = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=args.num_workers, pin_memory=True,
                                   drop_last=True,worker_init_fn=seed_worker,generator=g,shuffle=True if ((not args.distributed) and args.do_train_shuffle) else False)
    val_dataloader    = DataLoader(val_dataset  , args.valid_batch_size, sampler=val_datasampler,   num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return   train_dataset,   val_dataset, train_dataloader, val_dataloader


fourcast_default_step ={
    1:20,6: 20, 12: 10, 24:10
}
def get_test_dataset(args,test_dataset_tensor=None,test_record_load=None):
    time_step = args.Dataset.time_step if "fourcast" in args.Train.mode else fourcast_default_step[args.time_intervel] + args.Dataset.time_step
    dataset_kargs = copy.deepcopy(args.dataset_kargs)
    dataset_kargs['time_step'] = time_step
    if dataset_kargs['time_reverse_flag'] in ['only_forward','random_forward_backward']:
        dataset_kargs['time_reverse_flag'] = 'only_forward'
    dataset_type = eval(args.dataset_type) if isinstance(args.dataset_type,str) else args.dataset_type
    #print(dataset_kargs)
    split = args.split if hasattr(args,'split') and args.split else "test"
    test_dataset = dataset_type(split=split, with_idx=True,dataset_tensor=test_dataset_tensor,record_load_tensor=test_record_load,**dataset_kargs)
    if args.wrapper_model and hasattr(eval(args.wrapper_model),'pred_channel_for_next_stamp'):
        test_dataset.pred_channel_for_next_stamp = eval(args.wrapper_model).pred_channel_for_next_stamp
    if args.multi_branch_order is not None:
        test_dataset.multi_branch_order = args.multi_branch_order
    assert hasattr(test_dataset,'clim_tensor')
    test_datasampler  = DistributedSampler(test_dataset,  shuffle=False) if args.distributed else None
    test_dataloader   = DataLoader(test_dataset, args.valid_batch_size, sampler=test_datasampler, num_workers=args.num_workers, pin_memory=False)
    
    
    return   test_dataset,   test_dataloader

