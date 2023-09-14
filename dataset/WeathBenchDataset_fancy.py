from .WeathBenchDataset import *

class WeatherBenchSPnorm(WeatherBenchPhysical): #<--- SP means space time
    space_time_meanstds = None # ( will load it dynamically as a tensor (8784,110,32,64))

    def __init__(self, split="train", shared_dataset_tensor_buffer=(None,None), config = {}):
        super().__init__(split=split, shared_dataset_tensor_buffer=(None, None), config=config)
        print("""
        =======================================================================================================
            If you want to use the space time means and stds, notice the data will get two type of normlization.
              - x = (x - mean)/std. This is the general normlization.
              - x_t =  (x_t - mean_t) / std_d. This is the per timestamp normlization.
            The behavior of how to normlize in data is depend on how you calculate the mean_t and std_t.
                The coded method is for 
            Thus, recover original data from the runtime data flow is messy. Single normlizer is not enough.
        =======================================================================================================
              """)
        use_offline_data = config.get('use_offline_data', 0)
        assert use_offline_data == 0, "WeatherBench32x64SPnorm do not support use_offline_data"
        self.mean, self.std = self.mean_std[:,self.channel_choice]  # (2, P, W, H)
        
    def get_config_from_preset_pool(self, dataset_flag):
        _component_list68 = self._component_list68
        _component_list69 = _component_list68 + [4] # add total precipitation
        if   dataset_flag == '2D68S':return _component_list68 , TimewiseNormlizer()
        elif dataset_flag == '2D69S':return _component_list69,  TimewiseNormlizer()
        else:
            raise NotImplementedError(f"dataset_flag {dataset_flag} is not implemented")
        
    def make_data_physical_reasonable(self, data, odata, idx=None):
        data     = (data - self.mean[None])/self.std[None]
        mean,std = self.get_space_time_mean_std(idx)
        data     = (data - mean)/(std)
        return data
    
    def get_space_time_mean_std(self,idx):
        if self.space_time_meanstds is None:
            self.space_time_meanstds = np.zeros((8784,2,110,self.resolution_h,self.resolution_w))
            self.loaded_flag = {}
        now_time_stamp    = self.datatimelist_pool[self.split][idx]
        start_time_stamp  = np.datetime64("2016-01-01")
        the_meanstd_index = int((now_time_stamp - start_time_stamp)/ np.timedelta64(1,'h'))
        the_meanstd_index = the_meanstd_index%8784
        if the_meanstd_index not in self.loaded_flag:
            self.loaded_flag[the_meanstd_index]=1
            self.space_time_meanstds[the_meanstd_index] = np.load(f"datasets/weatherbench32x64/sp_meanstd/{the_meanstd_index:4d}.npy") # this will gradually take 28G memory
        return self.space_time_meanstds[the_meanstd_index][:,self.channel_choice] #(2, Channel, 32, 64)

    def recovery(self, x, indexes):
        real_meanstds = torch.from_numpy(np.stack([self.get_space_time_mean_std(idx) for idx in indexes], 1))  # (2,B, 68, 32, 64)
        real_mean, real_std = real_meanstds
        x = x * real_std.to(x.device) + real_mean.to(x.device) # (B,68,32,64) * (B,68,32,64) + (B,68,32,64)
        x = x * torch.from_numpy(self.std[None]).to(x.device) + torch.from_numpy(self.mean[None]).to(x.device)
        return x

    @staticmethod
    def unique_name(config):
        return f"WeatherBench_SpaceTime_{config.resolution_h}x{config.resolution_w}_{config.dataset_flag}"
   
class WeatherBenchPhysical_H5(WeatherBenchPhysical): #<--- TODO: should remove, and move HDF5 file reading into the load_otensor

    def load_otensor(self,idx):
        if not hasattr(self, 'h5pool'): 
            self.h5pool= {}
            self.hour_map={}
            for i,(year,hour) in enumerate(self.single_data_path_list):
                if year not in self.hour_map:self.hour_map[year]={}
                self.hour_map[year][hour]=len(self.hour_map[year])
        year, hour = self.single_data_path_list[idx]
        if year not in self.h5pool:
            self.h5pool[year] = h5py.File(f"datasets/weatherbench_h5/{year}.hdf5", 'r')['data']
        odata = self.h5pool[year][self.hour_map[year][hour]]
        return odata

class WeatherBenchSelf(WeatherBench):
    """
    This dataset deal with in-timestamp prediction that want to use
    - some of the channel of timestamp T 
    to predict 
    - some of the channel of timestamp T
    
    """
    def __init__(self, split="train", shared_dataset_tensor_buffer=(None, None), config={}):
        super().__init__(split=split, shared_dataset_tensor_buffer=shared_dataset_tensor_buffer, config=config)
        raise NotImplementedError("TODO: need to modify the code to make it work")
        picked_input_property  = config.in_timestamp_input_channel
        picked_output_property = config.in_timestamp_output_channel
        assert picked_input_property
        assert picked_output_property
        if isinstance(picked_input_property, int):picked_input_property   = [picked_input_property]
        if isinstance(picked_output_property, int):picked_output_property = [picked_output_property]
        assert len(set(picked_input_property) & set(picked_output_property)) == 0
        
        
        ### origin use physical channel index, the better way is use property name directly and convert to channel index
        # self.picked_input_channel = []
        # for p in picked_input_property:
        #     self.picked_input_channel += list(range(p*14, (p+1)*14))

        # self.picked_output_channel = []
        # for p in picked_output_property:
        #     self.picked_output_channel += list(range(p*14, (p+1)*14))

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        batch = self.get_item(idx)
        batch = [{'field':batch[self.picked_input_channel]}, {'field':batch[self.picked_output_channel]}]
        return batch if not self.with_idx else (idx, batch)


class WeatherBenchdeseasonal(WeatherBench):
    """
    Seasonal Dataset manully split a pre-computed seasonal pattern and only forward its this noise pattern
    """ 
    def __init__(self, split="train", shared_dataset_tensor_buffer=(None, None), config={}):
        super().__init__(split=split, shared_dataset_tensor_buffer=shared_dataset_tensor_buffer, config=config)
        assert self.use_offline_data == 2
        assert self.use_time_stamp
        self.deseasonal_mean, self.deseasonal_std = torch.from_numpy(self.load_numpy_from_url(
                            os.path.join(self.root+"_offline", "mean_stds_deseasonal.npy"))).reshape(2, 70, 1, 1)
        self.seasonal_tensor = torch.Tensor(np.load(os.path.join(self.root+"_offline", "seasonal1461.npy")))
        time_stamps          = self.datatimelist_pool[self.split]
        sean_start_stamps    = np.datetime64("1979-01-02")
        offset               = ((time_stamps - sean_start_stamps) % (1461*np.timedelta64(6, "h")))//np.timedelta64(6, "h")
        self.timestamp       = torch.LongTensor(offset)

    def addseasonal(self, tensor, time_stamps_offset):
        if self.seasonal_tensor.device != tensor.device:
            self.deseasonal_std_tensor  = self.deseasonal_std_tensor.to(tensor.device)
            self.deseasonal_mean_tensor = self.deseasonal_mean_tensor.to(tensor.device)
            self.seasonal_tensor        = self.seasonal_tensor.to(tensor.device)
        tensor = tensor*self.deseasonal_std_tensor + self.deseasonal_mean_tensor
        # (B, 70, 32, 64)
        tensor = tensor + self.seasonal_tensor[time_stamps_offset.long()]
        return tensor

    def recovery(self, tensor, time_stamps_offset):
        return self.addseasonal(tensor, time_stamps_offset)

    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:
            root = WeatherBench.default_root
        if use_offline_data:
            dataset_flag = kargs.get('dataset_flag')
            data_name = f"{split}_{dataset_flag}_deseasonal.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root+'_offline', data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor, record_load_tensor


class WeathBench68pixelnorm(WeatherBench):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        assert self.use_offline_data == 2
        assert self.dataset_flag == '2D68K'
        self.pixelnorm_mean, self.pixelnorm_std = self.load_numpy_from_url(os.path.join(self.root+"_offline", "means_stds_pixelnorm.npy"))
        self.pixelnorm_mean_tensor = torch.Tensor(self.pixelnorm_mean).reshape(1, 68, 32, 64)
        self.pixelnorm_std_tensor  = torch.Tensor(self.pixelnorm_std).reshape(1, 68, 32, 64)

    def recovery(self, tensor):
        if self.pixelnorm_mean_tensor.device != tensor.device:
            self.pixelnorm_std_tensor = self.pixelnorm_std_tensor.to(tensor.device)
            self.pixelnorm_mean_tensor = self.pixelnorm_mean_tensor.to(tensor.device)
        tensor = tensor*self.pixelnorm_std_tensor + self.pixelnorm_mean_tensor
        return tensor

    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:
            root = WeatherBench.default_root
        if use_offline_data:
            dataset_flag = kargs.get('dataset_flag')
            data_name = f"{split}_{dataset_flag}_pixelnorm.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root+'_offline', data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor, record_load_tensor


class WeathBench55withoutH(WeatherBench):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        assert self.use_offline_data == 2
        assert self.dataset_flag == '2D55N'
        self.dataset_tensor = self.dataset_tensor[:, :55]
        #print(self.dataset_tensor.shape)

    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:
            root = WeatherBench.default_root
        if use_offline_data:
            dataset_flag = "2D70N"
            data_name = f"{split}_{dataset_flag}.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root, data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor, record_load_tensor


class WeatherBenchDeltaDataset(WeatherBench):
    """
    delta model deal with the delta of the data, so we need to load 
    1. the mean and std of the delta.
    2. the original data
    """
    def __init__(self, **kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        self.delta_mean, self.delta_std = self.load_numpy_from_url(os.path.join(self.root, "delta_mean_std.npy"))
        self.delta_mean = self.delta_mean.reshape(70, 1, 1)
        self.delta_std = self.delta_std.reshape(70, 1, 1)
        self.delta_mean_tensor = torch.Tensor(self.delta_mean).reshape(1, 70, 1, 1)
        self.delta_std_tensor = torch.Tensor(self.delta_std).reshape(1, 70, 1, 1)

    def __len__(self):
        return len(self.dataset_tensor) - self.time_step*self.time_intervel + 1 - 1

    def combine_base_delta(self, base, delta):
        if self.delta_std_tensor.device != delta.device:
            self.delta_std_tensor = self.delta_std_tensor.to(delta.device)
        return base + delta*self.delta_std_tensor + self.delta_mean_tensor

    def recovery(self, base, delta):
        return self.combine_base_delta(base, delta)

    def get_item(self, idx, reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        assert self.use_offline_data
        data = self.dataset_tensor[idx]
        delta = (self.dataset_tensor[idx + 1] - data)
        delta = (delta - self.delta_mean)/self.delta_std
        return data, delta


class WeatherBenchUpSize_64_to_128:
    """
    This dataset will contain both 32x64 and 64x128 data
    """
    def __init__(self, dataset_tensor_1=None,record_load_tensor_1=None,dataset_tensor_2=None,record_load_tensor_2=None,**kargs):
        assert kargs['root']
        newkargs = copy.deepcopy(kargs)
        newkargs['split']='subtrain'
        newkargs['root'] = kargs['root'].replace('32x64','64x128').replace('128x256','64x128')
        
        self.dataset_64x128 = WeatherBench64x128CK(dataset_tensor=dataset_tensor_1,record_load_tensor=record_load_tensor_1,**newkargs)
        newkargs['root'] = kargs['root'].replace('32x64','128x256').replace('64x128','128x256')
        self.dataset_128x256 = WeatherBench64x128CK(dataset_tensor=dataset_tensor_2,record_load_tensor=record_load_tensor_2,**newkargs)
        self.use_offline_data = False
    def __getitem__(self,idx):            
        return self.dataset_64x128.get_item(idx), self.dataset_128x256.get_item(idx)
    def __len__(self):
        return len(self.dataset_64x128)

    @staticmethod
    def create_offline_dataset_templete(split='train', years=None, root=None, do_in_class=False, **kargs):
        # when calling, use 'train' represent  64x128 branch
        # when calling, use 'valid' represent 128x256 branch
        if do_in_class:return None, None
        years = WeatherBench64x128.years_split['subtrain']
        batches = len(WeatherBench64x128.init_file_list(years))
        if split == 'train':
            return torch.empty(batches, 69, 64, 128), torch.zeros(batches)
        elif split == 'valid':
            return torch.empty(batches, 69, 128, 256), torch.zeros(batches)
        else:
            raise