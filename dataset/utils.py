class NanDetect:
    def __init__(self, logsys, use_amp):
        self.logsys = logsys
        self.nan_count = 0
        self.good_loss_count = 0
        self.default_amp = use_amp
        self.try_upgrade_times = 0

    def nan_diagnose_weight(self, model, loss, scaler):
        skip = torch.zeros_like(loss)
        downgrad_use_amp = torch.zeros_like(loss)
        upgrade_use_amp = torch.zeros_like(loss)
        nan_count = self.nan_count
        logsys = self.logsys
        if torch.isnan(loss):
            self.good_loss_count = 0
            # we will check whether weight has nan
            bad_weight_name = []
            bad_check = False
            for name, p in model.named_parameters():
                if torch.isnan(p).any():
                    bad_check = True
                    bad_weight_name.append(name)
            if bad_check:
                logsys.info(f"the value is nan in weight:{bad_weight_name}")
                raise NotImplementedError  # optuna.TrialPruned()
            else:
                nan_count += 1
                if nan_count > 5 and model.use_amp:
                    downgrad_use_amp += 1
                if nan_count > 10:
                    logsys.info("too many nan happened")
                    raise NotImplementedError  # optuna.TrialPruned()
                logsys.info(
                    f"detect nan, now at {nan_count}/10 warning level, pass....")
                skip += 1
            self.good_loss_count = 0
        else:
            self.good_loss_count += 1
            if self.good_loss_count > 5 and self.default_amp and self.try_upgrade_times < 5 and not model.use_amp:
                upgrade_use_amp += 1
                self.try_upgrade_times += 1
            nan_count = 0
        self.nan_count = nan_count
        if hasattr(model, 'module'):
            dist.all_reduce(skip)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            dist.all_reduce(downgrad_use_amp)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            # 0*0*0*0 = 0; 1*1**1 =1;
            dist.all_reduce(upgrade_use_amp,
                            torch.distributed.ReduceOp.PRODUCT)
        if downgrad_use_amp:
            logsys.info(
                f"detect nan loss during training too many times and we are now at `autograd.amp` mode. so we will turn off amp mode ")
            if model.use_amp:
                model.use_amp = False
                scaler._enabled = False
        elif upgrade_use_amp:
            logsys.info(
                f"detect nan loss during training too many times and we are now at `autograd.amp` mode. so we will turn off amp mode ")
            if model.use_amp:
                model.use_amp = True
                scaler._enabled = True  # notice the default scaler should be activated at initial
        return skip

    def nan_diagnose_grad(self, model, loss, scaler):
        skip = torch.zeros_like(loss)
        downgrad_use_amp = torch.zeros_like(loss)
        logsys = self.logsys
        nan_count = self.nan_count
        # we will check whether weight has nan
        bad_weight_name = []
        bad_check = False
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                bad_check = True
                bad_weight_name.append(name)
        if bad_check:
            logsys.info(f"the value is nan in weight.grad:{bad_weight_name}")
            nan_count += 1
            if nan_count > 10:
                logsys.info("too many nan happened")
                raise
            logsys.info(
                f"detect nan, now at {nan_count}/10 warning level, pass....")
            skip += 1

        if hasattr(model, 'module'):
            dist.all_reduce(skip)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
            dist.all_reduce(downgrad_use_amp)  # 0+0+0+0 = 0; 0 + 1 + 0 + 1 =1;
        if downgrad_use_amp:
            if model.use_amp:
                model.use_amp = False
                scaler._enabled = False
                model.use_amp = bool(downgrad_use_amp.item()) and model.use_amp
        return skip

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
    
    train_datasampler = DistributedSampler(train_dataset, shuffle=args.do_train_shuffle, seed=args.seed) if args.distributed else None
    val_datasampler   = DistributedSampler(val_dataset,   shuffle=False) if args.distributed else None
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader  = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=args.num_workers, pin_memory=True,
                                   drop_last=True,worker_init_fn=seed_worker,generator=g,shuffle=True if ((not args.distributed) and args.do_train_shuffle) else False)
    val_dataloader    = DataLoader(val_dataset  , args.valid_batch_size, sampler=val_datasampler,   num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return   train_dataset,   val_dataset, train_dataloader, val_dataloader


fourcast_default_step ={
    1:20,6: 20, 12: 10, 24:10
}
def get_test_dataset(args,test_dataset_tensor=None,test_record_load=None):
    time_step = args.time_step if "fourcast" in args.mode else fourcast_default_step[args.time_intervel] + args.time_step
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
