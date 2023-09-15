from .base import BaseDataset
import numpy as np 
import os
import torch
import pandas as pd
import copy
from utils.tools import get_sub_luna_point,get_sub_sun_point
from utils.timefeatures import time_features
from .normlizer import PreGauessNormlizer,PreUnitNormlizer,NoneNormlizer,PostGauessNormlizer,PostUnitNormlizer,TimewiseNormlizer
from tqdm.auto import tqdm 
from utils.tools import get_center_around_indexes, get_center_around_indexes_3D
import json
from .utils import load_numpy_from_url

def get_init_file_list(dates):
    """
    ---> We will require the dates via np.datetime64 format
    ---> Old code use year list below
    # file_list = []
    # for year in years:
    #     if year == 1979: # 1979年数据只有8753个，缺少第一天前7小时数据，所以这里我们从第二天开始算起
    #         for hour in range(17, 8753, 1):
    #             file_list.append([year, hour])
    #     else:
    #         if year % 4 == 0:
    #             max_item = 8784
    #         else:
    #             max_item = 8760
    #         for hour in range(0, max_item, 1):
    #             file_list.append([year, hour])
    # return file_list
    """
    years          = dates.astype('datetime64[Y]').astype(int)   ## start from 1970, thus 1979 is 9
    start_of_years = years.astype('datetime64[Y]')
    hours          = (dates - start_of_years) / np.timedelta64(1, 'h')
    hours[years == 9] -= 7 # 1979 data lake 7 hours
    hours = np.maximum(hours, 0)  # ensure no negative hour values
    file_list      = np.vstack((years+1970, hours)).T.astype(int)
    return file_list


def get_default_timestamp_pool(config):
    time_unit = config.get('time_unit', 1 )
    return {'train':np.arange(np.datetime64("1979-01-02"), np.datetime64("2017-01-01"), np.timedelta64(time_unit, "h")),
            'valid':np.arange(np.datetime64("2017-01-01"), np.datetime64("2018-01-01"), np.timedelta64(time_unit, "h")),
            'test' :np.arange(np.datetime64("2018-01-01"), np.datetime64("2019-01-01"), np.timedelta64(time_unit, "h")),
            'ftest':np.arange(np.datetime64("1979-01-02"), np.datetime64("1980-01-01"), np.timedelta64(time_unit, "h")),
            }

def read_channel_list_file(file_path):
    with open(file_path, 'r') as f:
        channel_list = json.load(f)
    return channel_list

def get_timestamp_date(split, config):
    timestamps  = config.get('timestamps_list', None)
    defaultPool = get_default_timestamp_pool(config)
    time_unit   = config.get('time_unit', 1 )
    if timestamps is None:
        timestamps = defaultPool[split]
        print(f"you don't assign a timestamp list for {split}, then we use the default timestamp config  per {time_unit} hour")
    else:
        print(f"noice you assign your own year timestamps list from {timestamps.min} to {timestamps.max()}" )
    return timestamps  # [[year,idx],[year,idx],.....]

def get_vnames_from_config(config):
    channel_name_list = config.get('channel_name_list')
    if isinstance(channel_name_list, str):
        channel_name_list = read_channel_list_file(channel_name_list)
    elif isinstance(channel_name_list, list):
        channel_name_list = channel_name_list
    else:
        raise NotImplementedError(f"channel_name_list {channel_name_list} must be filepath or list of channel name")
    return channel_name_list

def preload_full_dataset(split, config):
    '''
    This will load the whole dataset into memory, it will be very slow and consume a lot of memory
    '''
    
    root           = config.get('root')
    ### first check whether the train.npy/test.npy/valid.npy is exist
    fast_offline_data_name = os.path.join(root, f"{split}.npy")
    if os.path.exists(fast_offline_data_name):
        print(f"detect fast offline data in {fast_offline_data_name}, use this")
        dataset_tensor     = torch.Tensor(np.load(fast_offline_data_name))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor
    else:
        init_file_list = get_init_file_list(get_timestamp_date(split, config))
        print(f"dont find offline data in {fast_offline_data_name}")
        print("then seek split file store in format {root}/{year}/{year}-{hour:04d}.npy")
        loaded_tensor  = None
        for i, (year, hour) in tqdm(enumerate(init_file_list)):
            url = f"{root}/{year}/{year}-{hour:04d}.npy"
            odata = load_numpy_from_url(None,url)
            if loaded_tensor is None:
                assert len(init_file_list)*np.prod(odata.shape) < 2**32, "The dataset is too large, we can't load it into memory"
                loaded_tensor = torch.empty(len(init_file_list), *odata.shape)
            loaded_tensor[i] = torch.from_numpy(odata)
        print("concat!")
        loaded_tensor  = torch.stack(loaded_tensor) 
        loaded_flag    = torch.ones(len(loaded_tensor)) 
        print("finish preload the whole dataset into memory")
        return loaded_tensor, loaded_flag

def get_the_dataset_flag(config):
    channel_picked_name = os.path.split(config.get('channel_name_list'))[-1].split('.')[0]
    normlizer_name      = config.get('normlized_flag','N')
    return channel_picked_name+normlizer_name

class WeatherBenchBase(BaseDataset):
    time_unit  = 1
        
    single_vnames = [ "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_cloud_cover",
                  "total_precipitation", "toa_incident_solar_radiation"]
    level_vnames= []
    for physics_name in ["geopotential", "temperature",
                            "specific_humidity","relative_humidity",
                            "u_component_of_wind","v_component_of_wind",
                            "vorticity","potential_vorticity"]:
        for pressure_level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
            level_vnames.append(f"{pressure_level}hPa_{physics_name}")

    weatherbench_property_name_list = single_vnames + level_vnames
    ## the 110 channel for weatherbench dataset
    cons_vnames= ["lsm", "slt", "orography"]

class WeatherBench(WeatherBenchBase):
    '''
    In-memory dataset, if we activate the in memory procedure, please create shared tensor 
    via Tensor.share_memory_() before call this module and pass the create tensor as args of this module
    the self.dataset_tensor should be same shape as the otensor, for example (10000, 110, 32, 64)
    the self.record_load_tensor is used to record which row is record.
    This implement will automatively offline otensor after load it once.

    no matter we use in-memory technology or not, we always initialized two tensor,
    one is the whole tensor dataset stored in (B, P, W, H) tensor 
    one is the loadQ tensor that judge whether the real data is loaded.
    '''
    default_root = None
    reduce_Field_coef = np.array([1])
    clim_tensor = [0]
    img_shape = (32, 64)
    record_load_tensor = None
    dataset_tensor = None

    def __init__(self, split="train", shared_dataset_tensor_buffer=(None, None),with_idx=False, config={}, debug=False):
        self.root = config.get('root', self.default_root)
        assert self.root is not None
        print(f"use dataset in {self.root} for {split}")
        self.split = split
        self.with_idx=with_idx
        self.timestamp_date=get_timestamp_date(split, config)
        self.single_data_path_list = get_init_file_list(self.timestamp_date)
        self.time_step     = config.get('time_step', 2)
        self.time_intervel = config.get('time_intervel', 1)
        self.resolution_w  = config.get('resolution_w', 1440)
        self.resolution_h  = config.get('resolution_h',  720)
        
        self.normlized_flag = config.get('normlized_flag', 'N')
        self.use_time_feature = config.get('use_time_feature', False)
        self.time_reverse_flag = config.get('time_reverse_flag', 'only_forward')
        self.add_LunaSolarDirectly = config.get('add_LunaSolarDirectly', False)
        self.offline_data_is_already_normed = config.get('offline_data_is_already_normed', False)

        self.constant_channel_pick = config.get('constant_channel_pick', None)
        self.make_data_physical_reasonable_mode = config.get('make_data_physical_reasonable_mode', None)
        
        
        self.patch_range  = patch_range = config.get('patch_range', None)
        self.cross_sample = config.get('cross_sample', True) and ((self.split == 'train') or debug) and (self.patch_range is not None)
        
        #self.vnames= [self.weatherbench_property_name_list[i] for i in self.channel_choice]
        self.dataset_flag = get_the_dataset_flag(config)
        
        self.vnames = get_vnames_from_config(config)
        self.channel_choice = np.array(self.get_channel_choice(self.vnames))

        self.mean_std = self.load_original_meanstds()  # (2, P, W, H)
        self.normlizer = self.get_normlizer(
            self.mean_std[:, self.channel_choice][..., None, None], self.normlized_flag)

        #self.channel_choice, self.normlizer = self.get_config_from_preset_pool(self.dataset_flag)
        self.mean, self.std = self.normlizer.mean, self.normlizer.std
        self.unit_list      = self.std

        self.constants = torch.from_numpy(self.load_numpy_from_url(
            os.path.join(self.root, "constants.npy")))

        assert ((shared_dataset_tensor_buffer[0] is None and shared_dataset_tensor_buffer[1] is None) or
                (shared_dataset_tensor_buffer[0] is not None and shared_dataset_tensor_buffer[1] is not None))

        if shared_dataset_tensor_buffer[0] is None:
            self.dataset_tensor, self.record_load_tensor = self.create_offline_dataset_templete(
                split, create_buffer=False, config=config)
        else:
            self.dataset_tensor, self.record_load_tensor = shared_dataset_tensor_buffer
        if self.dataset_tensor is not None:
            assert len(self.dataset_tensor) == len(self.timestamp_date), f"{len(self.dataset_tensor)} == {len(self.timestamp_date)}"
            assert self.dataset_tensor.shape[-2] == self.resolution_h
            assert self.dataset_tensor.shape[-1] == self.resolution_w
        if self.single_data_path_list is not None:
            assert len(self.single_data_path_list) == len(self.timestamp_date)
        
        

        if self.time_reverse_flag: ## should add this before change vnames
            self.volicity_idx = [i for i, name in enumerate(self.vnames) if (
                ('v_component' in name) or ('u_component' in name))]

        # if self.constant_channel_pick: ### no need change vnames anymore, since we add constant at another key
        #     self.vnames = self.vnames + [self.cons_vnames[i] for i in self.constant_channel_pick]
            
        if self.use_time_feature:
            self.timestamp_date = time_features(pd.to_datetime(self.timestamp_date)).transpose(1, 0)

        if self.add_LunaSolarDirectly:
            self.LaLotude, self.LaLotudeVector = self.get_mesh_lon_lat(
                self.resolution_h, self.resolution_w)  # (32, 64 )

        
        self.img_shape    = (self.resolution_h, self.resolution_w)  
        if self.patch_range:
            if len(self.channel_choice.shape) == 2:
                patch_range = patch_range if isinstance(patch_range, (list, tuple)) else (patch_range, patch_range, patch_range)
                self.center_index, self.around_index = get_center_around_indexes_3D(patch_range, self.img_shape)
            else:
                patch_range = patch_range if isinstance(patch_range, (list, tuple)) else (patch_range, patch_range)
                self.center_index, self.around_index = get_center_around_indexes(patch_range, self.img_shape)
            self.patch_range = patch_range
            print(f"notice we will use around_index{self.around_index.shape} to patch data")

    def get_channel_choice(self, property_names):
        if isinstance(property_names, (tuple,list)):
            return [self.get_channel_choice(property_name) for property_name in property_names]
        elif isinstance(property_names, str):
            return self.weatherbench_property_name_list.index(property_names) 
        else:
            raise NotImplementedError(f"property_names {property_names} must be str or list of str")
        
    def load_original_meanstds(self):
        mean_std_path = os.path.join(self.root, "mean_std.npy")
        if os.path.exists(mean_std_path):
            return self.load_numpy_from_url(mean_std_path)  # (2, P, W, H)
        else:
            print("""
                Warning!!!! The mean std file is not founed, we will use the mean=0 and std=1
                  """)
            return 0, 1

    @staticmethod
    def unique_name(config):
        return f"WeatherBench_{config.get('resolution_h')}x{config.get('resolution_w')}_{get_the_dataset_flag(config)}"


    @staticmethod
    def _add_constant_mean_std(mean_std, channel_num):
        mean, std = mean_std  # (70,1,1)
        mean = np.concatenate([mean, np.zeros(channel_num, *mean.shape[1:])])
        std = np.concatenate([std,  np.ones(channel_num, *std.shape[1:])])
        return np.stack([mean, std])  # (2, 71, 1, 1)

    @staticmethod
    def get_normlizer(mean_std, normlized_flag):
        
        if normlized_flag[-1] == 'N':
            return PreGauessNormlizer(mean_std)
        elif normlized_flag[-1] == 'U':
            return PreUnitNormlizer(mean_std)
        elif normlized_flag[-1] == 'O':
            return PostGauessNormlizer(mean_std)
        else:
            raise NotImplementedError(
                f"normlized_flag {normlized_flag} is not implemented")

    def get_config_from_preset_pool(self, dataset_flag):
        if dataset_flag == '2D110N':
            return list(range(0, 110)), PreGauessNormlizer(self.mean_std.reshape(2, 110, 1, 1))
        elif dataset_flag == '2D110U':
            return list(range(0, 110)),   PreUnitNormlizer(self.mean_std.reshape(2, 110, 1, 1))
        elif dataset_flag == '2D104N':
            return list(range(6, 110)), PreGauessNormlizer(self.mean_std[:, 6:].reshape(2, 104, 1, 1))
        elif dataset_flag == '2D104U':
            return list(range(6, 110)),   PreUnitNormlizer(self.mean_std[:, 6:].reshape(2, 104, 1, 1))
        elif dataset_flag == '3D104N':
            return self.threeD_channel, PreGauessNormlizer(self.mean_std[:, 6:].reshape(2, 8, 13, 1, 1).mean(2))
        elif dataset_flag == '3D104U':
            return self.threeD_channel,   PreUnitNormlizer(self.mean_std[:, 6:].reshape(2, 8, 13, 1, 1).mean(2))
        elif dataset_flag == '2D110O':
            return list(range(0, 110)),  PostGauessNormlizer(self.mean_std.reshape(2, 110, 1, 1))
        elif dataset_flag == '2D104O':
            return list(range(6, 110)),  PostGauessNormlizer(self.mean_std[:, 6:].reshape(2, 104, 1, 1))
        elif dataset_flag == '3D104O':
            return self.threeD_channel, PostGauessNormlizer(self.mean_std[:, 6:].reshape(2, 8, 13, 1, 1).mean(2))
        else:
            raise NotImplementedError(
                f"dataset_flag {dataset_flag} is not implemented")

    def __len__(self):
        return len(self.single_data_path_list) - self.time_step*self.time_intervel + 1

    @staticmethod
    def create_offline_dataset_templete(split='test', create_buffer=False, fully_loaded=False, config={}):
        fast_offline_data_name = os.path.join(config.get('root'), f"{split}.npy")
        if os.path.exists(fast_offline_data_name):
            create_buffer = fully_loaded = True
        if not create_buffer:return None, None
        if fully_loaded:
            return preload_full_dataset(split, config)
        else:
            ### will dynamic load during the iteration
            init_file_list = get_init_file_list(get_timestamp_date(split, config))
            vnames = get_vnames_from_config(config)
            batches = len(init_file_list)
            h  = config.get('resolution_h')
            w  = config.get('resolution_w')
            ### this create the whole buffer
            assert batches*len(vnames)*h*w < 2**32, "The dataset is too large, we can't load it into memory"
            return (torch.empty(batches, len(vnames), h , w),
                    torch.zeros(batches))

    def load_otensor(self, idx):
        '''
        This will load the whole tensor of timestamp into memory. For very large snapshot, it may be inefficient.
        '''
        if (self.record_load_tensor is None) or (not self.record_load_tensor[idx]):
            year, hour = self.single_data_path_list[idx]
            url = f"{self.root}/{year}/{year}-{hour:04d}.npy"
            odata = self.load_numpy_from_url(url)
            if self.record_load_tensor is not None:
                '''
                <--- load data into shared buffer when dataset iteration
                '''
                self.record_load_tensor[idx] = 1
                self.dataset_tensor[idx] = torch.Tensor(odata)

        if self.record_load_tensor is not None:
            # .clone() <--- be careful that do not inplace modify the source data
            return self.dataset_tensor[idx]
        else:
            return odata

    def generate_runtime_data(self, idx, reversed_part=False):
        odata = self.load_otensor(idx)
        assert odata.shape[-2] == self.resolution_h
        assert odata.shape[-1] == self.resolution_w
        if isinstance(odata, torch.Tensor) and isinstance(self.channel_choice, np.ndarray):
            self.channel_choice = torch.LongTensor(self.channel_choice)
        data = odata[self.channel_choice]  # (110, W,H )->(P,W,H)
        data = self.make_data_physical_reasonable(data, odata, idx=idx)
        if not self.offline_data_is_already_normed:
            data = self.normlizer.do_pre_normlize(data)
        #assert len(data) == len(self.vnames) - len(self.constant_channel_pick)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if reversed_part:  # the reversed_part should be
            data = data.clone() if isinstance(data, torch.Tensor) else data.copy()
            data[reversed_part] = -data[reversed_part]
        return data

    def get_item(self, idx, reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        # if self.use_offline_data:
        #     data = self.dataset_tensor[idx]
        # else:
        data = self.generate_runtime_data(idx, reversed_part=reversed_part)

        location = None
        if self.cross_sample:
            center_h = np.random.randint(
                self.img_shape[-2] - (self.patch_range[-2]//2)*2)
            center_w = np.random.randint(self.img_shape[-1])
            if len(self.channel_choice.shape) == 2:
                center_z = np.random.randint(
                    self.img_shape[-3] - (self.patch_range[-3]//2)*2)
                location = self.around_index[center_z, center_h, center_w]
                #location = self.center_index[:,center_z, center_h, center_w]
            else:
                location = self.around_index[center_h, center_w]
                #location= self.center_index[:, center_h, center_w]

        if location is not None:
            if len(self.channel_choice.shape) == 2:
                patch_idx_z, patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_z, patch_idx_h, patch_idx_w]
            else:
                patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_h, patch_idx_w]

        output = {'field': data}

        if self.use_time_feature:
            output['timestamp'] = self.timestamp_date[idx]
        if self.add_LunaSolarDirectly:
            output['sun_mask'] = self.get_sum_mask(idx)
            output['luna_mask'] = self.get_moon_mask(idx)
        if self.constant_channel_pick and len(self.constant_channel_pick) > 0:
            output['constant'] = self.constants[self.constant_channel_pick]
        if self.patch_range:
            output['location'] = location
        return output

    def get_sum_mask(self, idx):
        timenow = self.timestamp_date[idx]
        sun_lon, sun_lat = get_sub_sun_point(timenow.item())
        sun_vector = np.stack([np.cos(sun_lat/180*np.pi)*np.cos(sun_lon/180*np.pi),
                               np.cos(sun_lat/180*np.pi) *
                               np.sin(sun_lon/180*np.pi),
                               np.sin(sun_lat/180*np.pi)])

        sun_mask = (self.LaLotudeVector@sun_vector).reshape(1,
                                                            self.resolution_h, self.resolution_w)
        return sun_mask

    def get_moon_mask(self, idx):
        timenow = self.timestamp_date[idx]
        moon_lon, moon_lat = get_sub_luna_point(timenow.item())
        moon_vector = np.stack([np.cos(moon_lat/180*np.pi)*np.cos(moon_lon/180*np.pi),
                                np.cos(moon_lat/180*np.pi) *
                                np.sin(moon_lon/180*np.pi),
                                np.sin(moon_lat/180*np.pi)])
        moon_mask = (self.LaLotudeVector@moon_vector).reshape(1,
                                                              self.resolution_h, self.resolution_w)
        return moon_mask

    def make_data_physical_reasonable(self, data, odata, idx=None):
        '''
        make the data physical reasonable, for example, the total precipitation should be positive
        '''
        if self.make_data_physical_reasonable_mode is None:
            return data
        raise NotImplementedError

    @staticmethod
    def get_mesh_lon_lat(tH=32, tW=64):
        ### this is a general version
        resolution = tH
        assert tW == 2*tH
        theta_offset = (180/resolution/2)
        latitude = (np.linspace(0, 180, resolution+1) +
                    theta_offset)[:resolution]
        longitude = np.linspace(0, 360, 2*resolution+1)[:(2*resolution)]
        x, y = np.meshgrid(latitude, longitude)
        LaLotude = np.stack([y, x])/180*np.pi
        LaLotudeVector = np.stack([np.cos(LaLotude[1])*np.cos(LaLotude[0]), np.cos(
            LaLotude[1])*np.sin(LaLotude[0]), np.sin(LaLotude[1])], 2)
        return LaLotude, LaLotudeVector
        #### Below is 721x1440 origin data version
        # sH = 720
        # sW = 1440
        # steph = sH /float(tH)
        # stepw = sW /float(tW)
        # x     = np.arange(0, sW, stepw).astype('int')
        # y     = np.linspace(0, sH-1, tH, dtype='int')
        # x, y  = np.meshgrid(x, y)
        # latitude   = np.linspace(90,-90,721)
        # longitude  = np.linspace(0,360,1440)
        # LaLotude = np.stack([latitude[y],longitude[x]])/180*np.pi
        # LaLotudeVector = np.stack([np.cos(LaLotude[1])*np.cos(LaLotude[0]),np.cos(LaLotude[1])*np.sin(LaLotude[0]),np.sin(LaLotude[1])],2)
        # return LaLotude,LaLotudeVector

class WeatherBenchPhysical(WeatherBench):
    default_root='datasets/weatherbench'
    use_offline_data = False           
            
    def make_data_physical_reasonable(self, data, odata, idx=None):
        '''
        make the data physical reasonable, for example, the total precipitation should be positive
        '''
        if self.make_data_physical_reasonable_mode is None:
            return data
        eg = torch if isinstance(data, torch.Tensor) else np
        if self.make_data_physical_reasonable_mode == 'use_constant_patch_lack_of_groud_geopotential':
            assert '2D70' in self.dataset_flag
            data = data.clone() # <-- avoid inplace modify
            data[14*4-1]    = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
            total_precipitaiton = odata[4]
            data[14*5-1]    = total_precipitaiton
            return data 
        if self.make_data_physical_reasonable_mode == 'build_resonal_3D_tensor':
            assert '3D70' in self.dataset_flag
            raise NotImplementedError("TODO: notice the input tensor now is a 3D tensor, we need to modify the code to make it work")
            data[14*4-1] = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
            total_precipitaiton = odata[4]
            newdata = data[14*5-1].clone() if isinstance(data,torch.Tensor) else data[14*5-1].copy()
            newdata[total_precipitaiton>0] = 100
            newdata[data[14*5-1]>100]=data[14*5-1][data[14*5-1]>100]
            data[14*5-1] = newdata
            shape= data.shape
            data = data.reshape(5,14,*shape[-2:])
            return data

class WeatherBenchMultibranchRandom(WeatherBench):
    """
    This is a multibranch random fetcher, it will randomly select a time_intervel from multibranch_select.
    We will use outside random fetcher to call this module. 
    For evaluation, we will use all time_intervel in multibranch_select via call set_time_intervel
    """
    def __init__(self, split="train", shared_dataset_tensor_buffer=(None,None),config = {}):
        super().__init__(split=split, shared_dataset_tensor_buffer=(None, None), config=config)
        use_offline_data = config.get('use_offline_data', 0)
        time_intervel    = config.get('time_intervel', 1)
        assert time_intervel == 1
        assert use_offline_data == 0
        self.multibranch_select= [int(t) for t in config['multibranch_select']]
    
    def set_time_intervel(self, time_intervel):
        self.time_intervel = time_intervel

    def __getitem__(self, idx):
        reversed_part = self.do_time_reverse(idx)
        time_intervel = self.time_intervel
        if self.split=='train':
            time_intervel = np.random.choice(self.multibranch_select)
            print(f"""
        ========> We use random time_intervel:{time_intervel}. However, this line should not appear here if your are in Train ,pde. You need call a random fetcher outside
                  """ )
        time_step_list = [idx+i*time_intervel for i in range(self.time_step)]
        if reversed_part:time_step_list = time_step_list[::-1]
        batch = [self.get_item(i, reversed_part) for i in time_step_list]
        return batch if not self.with_idx else (idx, batch)
           
class WeatherBench7066(WeatherBenchPhysical):
    """
    config = {
        'root':"/mnt/data/ai4earth/zhangtianning/datasets/WeatherBench/weatherbench_6hour/",
        'channel_name_list':"configs/datasets/WeatherBench/2D70.channel_list.json",
        'time_reverse_flag':'only_forward',
        'offline_data_is_already_normed':True,
        'time_unit':6
    }
    
    """
    
class WeatherBenchPatchDataset(WeatherBench):
    """
    config = {
        'root':"/mnt/data/ai4earth/zhangtianning/datasets/WeatherBench/weatherbench_6hour/",
        'channel_name_list':"configs/datasets/WeatherBench/2D70.channel_list.json",
        'time_reverse_flag':'only_forward',
        'offline_data_is_already_normed':True,
        'time_unit':6
    }
    """
        

        

    

if __name__ == "__main__":
    pass
    # import sys
    # import time
    # from tqdm import tqdm
    # from multiprocessing import Pool
    # from petrel_client.client import Client
    # import petrel_client
    # dataset = WeatherBench64x128(split='test',root='weatherbench:s3://weatherbench/weatherbench64x128/npy',dataset_flag='2D68N')
    # for i in tqdm(range(0, 1000)):
    #     a = dataset[i][0]

    # def load_data_range(dataset, start, end):
    #     for i in tqdm(range(start, end)):
    #         _ = dataset[i]

    # def parallel_load_data(dataset):
    #     total_len = len(dataset)
    #     processor = 10
    #     range_list = np.linspace(0, total_len, processor+1)
    #     range_list = [int(a) for a in range_list]
    #     print(range_list)
    #     res = []
    #     p = Pool(processor)
    #     for i in range(processor):
    #         res.append(p.apply_async(load_data_range, args=(
    #             dataset, range_list[i], range_list[i+1])))
    #         print(str(i) + ' processor started !')
    #     p.close()
    #     p.join()

    # def check_range(dataset, start, end):
    #     fail_list = []
    #     for i in tqdm(range(start, end)):
    #         if i > len(dataset):
    #             continue
    #         now_time = time.time()
    #         try:
    #             _ = dataset[i]
    #         except:
    #             print(f"fail at {i}")
    #         fail_list.append(i)
    #         cost = time.time() - now_time
    #         #print(cost)
    #     return fail_list
    # dataset = ERA5CephDataset(split='train')
    # print(ERA5CephDataset.__name__)
    # print(dataset[0])
    
    # print(len(dataset[0]))
    # print(len(dataset[1]))


    raise
    dataset = ERA5CephDataset(split='valid')
    print(len(dataset))
  
    for idx in range(len(dataset)):
        print(dataset.get_url(idx))
    exit()
    processor = 10
    unit = len(dataset)//processor
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(
            check_range, args=(dataset, i*unit, (i+1)*unit)))
        #print(str(i) + ' processor started !')

    p.close()
    p.join()
