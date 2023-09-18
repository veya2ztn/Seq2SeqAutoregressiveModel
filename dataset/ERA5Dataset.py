import os
from .base import BaseDataset, Client
from functools import lru_cache
import numpy as np
from utils.timefeatures import time_features
import pandas as pd
import torch

class ERA5BaseDataset(BaseDataset):
    
    full_vnames = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
        '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
        '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
        '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
        '50h_geopotential',
        'total_column_water_vapour',
    ]###<--- the order should not be changed
    channelpick_preset = {'physics': [5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13],'normal':list(range(20))}
    volicity_idx = [0, 5, 9, 14 , 1, 6, 10, 15]
    time_unit=6
    full_vnames_short = [
        'u10', 'v10', 't2m', 'sp', 'msl',
        'u', 'v', 'z',
        't', 'u', 'v', 'z', 'r',
        't', 'u', 'v', 'z', 'r',
        'z',
        'tcwv'

    ]
    mean_std_ERA5_20={
     '10m_u_component_of_wind': {'mean': -0.08244494681317033, 'std': 5.522365507485557},
     '10m_v_component_of_wind': {'mean': 0.1878750926415015,'std': 4.753310696543225},
     '2m_temperature': {'mean': 278.45956231728695, 'std': 21.364880588971882},
     'surface_pressure': {'mean': 96659.29942439323, 'std': 9576.018310416932},
     'mean_sea_level_pressure': {'mean': 100967.95123832714, 'std': 1317.7139732375715},
     '1000h_u_component_of_wind': {'mean': -0.07095991227445357,'std': 6.114047410072003},
     '1000h_v_component_of_wind': {'mean': 0.18681402351519094,'std': 5.2976192016365395},
     '1000h_geopotential': {'mean': 745.1660079545572, 'std': 1059.9845164332398},
     '850h_temperature': {'mean': 274.58180069739996, 'std': 15.676612264642246},
     '850h_u_component_of_wind': {'mean': 1.3814626339353238,'std': 8.15774947680599},
     '850h_v_component_of_wind': {'mean': 0.14620261110086222, 'std': 6.264685056755958},
     '850h_geopotential': {'mean': 13758.085881283701, 'std': 1459.716048599048},
     '850h_relative_humidity': {'mean': 69.10668451159029,'std': 26.372462450169042},
     '500h_temperature': {'mean': 253.0042938610095, 'std': 13.083165107000779},
     '500h_u_component_of_wind': {'mean': 6.544056434884079,'std': 11.968355707300768},
     '500h_v_component_of_wind': {'mean': -0.02533006083801716,'std': 9.185543423555893},
     '500h_geopotential': {'mean': 54130.677758771395, 'std': 3352.2513738740745},
     '500h_relative_humidity': {'mean': 50.39295631304117,'std': 33.51025992204092},
     '50h_geopotential': {'mean': 199408.6871957199, 'std': 5885.661841412361},
     'total_column_water_vapour': {'mean': 18.389728930352515,'std': 16.47164306296514}
     }
    Years = {
        'train': range(1979, 2016),
        'valid': range(2016, 2018),
        'test': range(2018, 2022),
        'all': range(1979, 2022)
    }
    
    
class ERA5CephDataset(ERA5BaseDataset):
    default_root     = 'cluster3:s3://era5npy'
    def __init__(self, split="train", config = {}):
        self.root         = config.get('root', self.default_root)
        self.years        = config.get('years', None)
        if self.years is None:self.years=self.Years[split] 
        self.time_step    = config.get('time_step', 2)
        self.crop_coord   = config.get('crop_coord', None)
        self.channel_last = config.get('channel_last', None)
        self.with_idx     = config.get('with_idx', False)
        self.dataset_flag = config.get('dataset_flag', 'normal')

        self.file_list    = self.init_file_list()
        self.vnames = [self.full_vnames[t] for t in self.channelpick_preset[self.dataset_flag]]
        self.name = f"ERA5_{self.time_step}-step-task"

    def __len__(self):
        return len(self.file_list) - self.time_step + 1

    def init_file_list(self):
        file_list = []
        for year in self.years:
            if year % 4 == 0:
                max_item = 1464
            else:
                max_item = 1460
            for hour in range(max_item):
                file_list.append([year, hour])
        return file_list


    
    
    @lru_cache(maxsize=32)
    def get_item(self, idx, reversed_part=False):
        year, hour = self.file_list[idx]
        arrays = []
        for name in self.vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            array = self.load_numpy_from_url(url) #(721, 1440)
            array = array[np.newaxis, :, :]
            arrays.append(array)
        arrays = np.concatenate(arrays, axis=0)
        if self.crop_coord is not None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            arrays = arrays.transpose(1,2,0)

        return arrays

    def get_url(self,idx):
        year, hour = self.file_list[idx]
        for name in self.vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"


class ERA5CephSmallDataset(ERA5BaseDataset):
    default_root = './datasets/era5G32x64_set'
    dataset_file = {'train': "train_data.npy",'valid': "valid_data.npy",'test':  "test_data.npy"}
    clim_file = "time_means.npy"
    img_shape = (32,64)
    time_intervel = 6
    def __init__(self, split="train", shared_dataset_tensor_buffer=(None,None),
                 config = {}):
        
        self.root         = config.get('root', self.default_root)
        self.years        = config.get('years', None)
        self.time_step    = config.get('time_step', 2)
        self.crop_coord   = config.get('crop_coord', None)
        self.channel_last = config.get('channel_last', None)
        self.with_idx     = config.get('with_idx', False)
        self.dataset_flag = config.get('dataset_flag', 'normal')
        self.use_time_stamp = config.get('use_time_stamp', False)
        self.time_reverse_flag = config.get('time_reverse_flag', False)
        
        ### load the source data into memory or use the shared memory
        dataset_path = os.path.join(self.root, self.dataset_file[split])
        self.data, self.record_load_tensor = self.create_offline_dataset_templete(dataset_path) \
            if shared_dataset_tensor_buffer[0] is None else shared_dataset_tensor_buffer
        self.clim_tensor = np.load(os.path.join(self.root, self.clim_file))
    
        if self.use_time_stamp:
            if split == 'train':
                datatimelist  = np.arange(np.datetime64("1979-01-01"), np.datetime64("2016-01-01"), np.timedelta64(6, "h"))
            elif split == 'valid':
                datatimelist  = np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"), np.timedelta64(6, "h"))
            elif split == 'test':
                datatimelist  = np.arange(np.datetime64("2018-01-01"), np.datetime64("2022-01-01"), np.timedelta64(6, "h"))
            self.timestamp    = time_features(pd.to_datetime(datatimelist)).transpose(1, 0)

        ### reconstruct the tensor data via channel pick
        self.channel_pick = self.channelpick_preset[self.dataset_flag]
        self.vnames       = [self.full_vnames[t] for t in self.channel_pick]
        self.unit_list    = [self.mean_std_ERA5_20[name]['std'] for name in self.vnames]
        self.clim_tensor  = self.clim_tensor[:,self.channel_pick]
        
        if self.random_time_step:
            print("we are going use random step mode. in this mode, data sequence will randomly set 2-6 ")
    
    @staticmethod
    def create_offline_dataset_templete(dataset_path):
        print(f"loading dataset from {dataset_path}")
        data = torch.Tensor(np.load(dataset_path))
        record_load_tensor = torch.ones(len(data))
        return data,record_load_tensor

    def __len__(self):
        return len(self.data) - self.time_step*self.time_intervel + 1

    def get_item(self, idx,reversed_part=False):
        arrays = self.data[idx]
        if reversed_part:
            arrays = arrays.copy() if isinstance(arrays, np.ndarray) else arrays.clone()#it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        arrays = arrays[self.channel_pick]
        if self.crop_coord is not None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            arrays = arrays.transpose(1, 2, 0) if isinstance(arrays, np.ndarray) else arrays.permute(1,2,0)
        if self.use_time_stamp:
            return {'field': arrays, 'timestamp': self.timestamp[idx]}
        else:
            return arrays

from utils.tools import get_center_around_indexes, get_center_around_indexes_3D
class ERA5CephSmallPatchDataset(ERA5CephSmallDataset):
    def __init__(self,*args, config ={}, **kargs):
        super().__init__(*args, config=config, **kargs)
        assert self.crop_coord is None
        self.cross_sample = config.get('cross_sample', True) and (self.split == 'train')
        self.patch_range  = patch_range = kargs.get('patch_range', 5)
        self.center_index,self.around_index=get_center_around_indexes(self.patch_range,self.img_shape)

        
    def get_item(self, idx,reversed_part=False):
        arrays = self.data[idx]
        if reversed_part:
            arrays = arrays.clone()#it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        arrays = arrays[self.channel_pick]
        if self.cross_sample:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2) 
            center_w = np.random.randint(self.img_shape[-1])
            patch_idx_h,patch_idx_w = self.around_index[center_h,center_w]
            arrays = arrays[..., patch_idx_h, patch_idx_w]

        if self.channel_last:
            if isinstance(arrays,np.ndarray):
                arrays = arrays.transpose(1,2,0)
            else:
                arrays = arrays.permute(1,2,0)
        if self.use_time_stamp:
            return {'field': arrays, 'timestamp': self.timestamp[idx]}
        else:
            return arrays

def load_test_dataset_in_memory(years=[2018], root='cluster3:s3://era5npy', crop_coord=None, channel_last=True, vnames=[]):
    client = None
    file_list = []
    for year in years:
        if year % 4 == 0:
            max_item = 1464
        else:
            max_item = 1460
        for hour in range(max_item):
            file_list.append([year, hour])
    print("loading data!!!")
    #file_list = file_list[:40]
    data = torch.empty(len(file_list), 720, 1440, 20, pin_memory=True)
    #print(data.shape)
    for idx, (year, hour) in tqdm(enumerate(file_list)):
        arrays = []
        for name in vnames:
            url = f"{root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            if "s3://" in url:
                if client is None:
                    client = Client(conf_path="~/petreloss.conf")
                array = read_npy_from_ceph(client, url)
            else:
                array = read_npy_from_buffer(url)
           # array = array[np.newaxis, :, :]
            arrays.append(array)
        arrays = np.stack(arrays, axis=0)
        if crop_coord is not None:
            l, r, u, b = crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if channel_last:
            arrays = arrays.transpose(1, 2, 0)
        data[idx] = torch.from_numpy(arrays)
    #data= torch.from_numpy(np.stack(data))

    return data

class ERA5CephInMemoryDataset(ERA5CephDataset):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=[2018],dataset_tensor=None,
                class_name='ERA5Dataset', root='cluster3:s3://era5npy', ispretrain=True, crop_coord=None,time_step=None,
                with_idx=False,dataset_flag=False,**kargs):
        if dataset_tensor is None:
            self.data=load_test_dataset_in_memory(years=years,root=root,crop_coord=crop_coord,channel_last=channel_last)
        else:
            self.data= dataset_tensor
        self.vnames= self.full_vnames
        if dataset_flag=='physics':
            self.data = self.data[:,self.full_physics_index]
            self.vnames= [self.full_vnames[t] for t in self.full_physics_index]
        self.mode = mode
        self.channel_last = channel_last
        self.with_idx = with_idx
        self.error_path = []
        if self.mode == 'pretrain':
            self.time_step = 2
        elif self.mode == 'finetune':
            self.time_step = 3
        elif self.mode == 'free5':
            self.time_step = 5
        if time_step is not None:self.time_step=time_step
        print(f"use time step {time_step}")

    def __len__(self):
        return len(self.data) - self.time_step + 1

    def __getitem__(self,idx):
        batch = [self.data[idx+i] for i in range(self.time_step)]
        return batch if not self.with_idx else (idx,batch)

import h5py
class ERA5Tiny12_47_96(ERA5BaseDataset):
    default_root = 'datasets/ERA5/h5_set'
    time_intervel =1
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=[2018],dataset_tensor=None,
                class_name='ERA5Dataset', root=None, ispretrain=True, crop_coord=None,time_step=2,
                with_idx=False,dataset_flag='normal',time_reverse_flag='only_forward',**kargs):
        if root is None:root=self.default_root
        if dataset_tensor is None:
            self.Field_dx= h5py.File(os.path.join(root,f"Field_dx_{split}.h5"), 'r')['data']
            self.Field_dy= h5py.File(os.path.join(root,f"Field_dy_{split}.h5"), 'r')['data']
            self.Field   = h5py.File(os.path.join(root,f"all_data_{split}.h5"), 'r')['data']
            self.Dt      = 6*3600
        else:
            raise NotImplementedError
        self.dataset_flag = dataset_flag
        
        self.vnames= [self.full_vnames[t] for t in [5, 9,14 , 6,10,15  ,2, 8,13,7,11,16]]
        self.mode  = mode
        self.channel_last = channel_last
        self.with_idx = with_idx
        self.error_path = []
        self.clim_tensor= np.load(os.path.join(root,f"time_means.npy"))[...,1:-1,:].reshape(4,3,47,96)
        self.time_step = time_step


        self.Field_channel_mean    = np.array([2.7122362e+00,9.4288319e-02,2.6919699e+02,2.2904861e+04]).reshape(4,1,1,1)
        self.Field_channel_std     = np.array([9.5676870e+00,7.1177821e+00,2.0126169e+01,2.2861252e+04]).reshape(4,1,1,1)
        self.Field_Dt_channel_mean =    np.array([  -0.02293313,-0.04692488  ,0.02711264   ,7.51324121]).reshape(4,1,1,1)
        self.Field_Dt_channel_std  =  np.array([  8.82677214 , 8.78834556  ,3.96441518   ,526.15269219]).reshape(4,1,1,1)
        self.unit_list = [1,1,1,1]

        if dataset_flag=='normalized_data':
            self.coef =coef= np.array([1,1,1e1,1e4]).reshape(4,1,1,1)
            self.Field_channel_mean    = self.Field_channel_mean    / coef
            self.Field_channel_std     = self.Field_channel_std     / coef
            self.Field_Dt_channel_mean = self.Field_Dt_channel_mean / coef
            self.Field_Dt_channel_std  = self.Field_Dt_channel_std  / coef
            self.clim_tensor = self.clim_tensor/coef
            self.unit_list = coef

        if dataset_flag == 'reduce':
            self.reduce_Field_coef    = np.array([-0.008569032217562018, -0.09391911216803356, -0.05143135231455811, -0.10192078794347732]).reshape(4,1,1,1)
            self.reduce_Field_Dt_channel_mean = np.array([2.16268301e-04 ,4.40459576e-03,-1.36775636e-03,-7.66760608e-01]).reshape(4,1,1,1)
            self.reduce_Field_Dt_channel_std  =  np.array([  3.26819142  ,4.04325207 , 1.99422196 ,235.31884021]).reshape(4,1,1,1)
        else:
            self.reduce_Field_coef = np.array([1])

        self.Field_mean     = None
        self.Field_std      = None
        self.Field_Dt_mean  = None
        self.Field_Dt_std   = None

        self.volicity_idx   = ([0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2])# we always assume the first two channel is volecity
        self.time_reverse_flag = time_reverse_flag
        self.set_time_reverse(time_reverse_flag)

    def __len__(self):
        # in h5 realization the last entry seem will exceed the limit
        # so we omit the last +1
        return len(self.Field) - self.time_step*self.time_intervel 

    def do_normlize_data(self, batch):
        assert isinstance(batch,(list,tuple))
        # assume input batch is
        # [ timestamp_1:[Field,Field_Dt,(physics_part) ],
        #   timestamp_2:[Field,Field_Dt,(physics_part) ],
        #     ...............................................
        #   timestamp_n:[Field,Field_Dt,(physics_part) ]
        if self.Field_mean is None:
            self.Field_mean    = self.Field_channel_mean
            self.Field_std     = self.Field_channel_std
            self.Field_Dt_mean = self.reduce_Field_Dt_channel_mean if self.dataset_flag == 'reduce' else self.Field_Dt_channel_mean
            self.Field_Dt_std  = self.reduce_Field_Dt_channel_std if self.dataset_flag == 'reduce' else self.Field_Dt_channel_std
            device = batch[0][0].device
            self.Field_mean    = torch.Tensor( self.Field_mean    )[None].to(device)#(1,4,1,1,1)
            self.Field_std     = torch.Tensor( self.Field_std     )[None].to(device)#(1,4,1,1,1)
            self.Field_Dt_mean = torch.Tensor( self.Field_Dt_mean )[None].to(device)#(1,4,1,1,1)
            self.Field_Dt_std  = torch.Tensor( self.Field_Dt_std  )[None].to(device)#(1,4,1,1,1)
        if isinstance(batch[0],(list,tuple)):
            for data in batch:
                data[0] = (data[0] - self.Field_mean)/self.Field_std #Field
                if len(data)>1:
                    data[1] = (data[1] - self.Field_Dt_mean)/self.Field_Dt_std
            return batch
        else:
            return [(data - self.Field_mean)/self.Field_std for data in batch]

    def inv_normlize_data(self,batch):

        if self.Field_mean is None:raise NotImplementedError("please do forward normlize at least once")
        assert isinstance(batch,(list,tuple))
        if isinstance(batch[0],(list,tuple)):
            for data in batch:
                data[0] = data[0]*self.Field_std   + self.Field_mean #Field
                if len(data)>1:
                    data[1] = data[1]*self.Field_Dt_std+ self.Field_Dt_mean
            return batch
        else:
            return [data*self.Field_std + self.Field_mean for data in batch]

    def get_item(self,idx,reversed_part=False):
        now_Field = self.Field[idx]
        next_Field= self.Field[idx+1]
        if reversed_part:
            now_Field = now_Field.copy()
            next_Field= next_Field.copy()
            now_Field[reversed_part] = -now_Field[reversed_part]
            next_Field[reversed_part]= -next_Field[reversed_part]

        if self.dataset_flag == 'only_Field':
            Field        = now_Field[...,1:-1,:]#(P,3,47,96)
            return Field
        elif self.dataset_flag == 'normalized_data':
            Field        = now_Field[...,1:-1,:]/self.coef
            return Field
        else:
            Field        = now_Field[...,1:-1,:]#(P,3,47,96)
            NextField    = next_Field[...,1:-1,:]#(P,3,47,96)
            u            = Field[0:1]
            v            = Field[1:2]
            T            = Field[2:3]
            p            = Field[3:4]
            Field_dt     = NextField - Field #(P,3,47,96)
            Field_dx     = self.Field_dx[idx]#(P,3,47,96)
            Field_dy     = self.Field_dy[idx]#(P,3,47,96)
            if reversed_part:
                Field_dx=Field_dx.copy()
                Field_dy=Field_dy.copy()
                Field_dx[reversed_part] = -Field_dx[reversed_part]
                Field_dy[reversed_part] = -Field_dy[reversed_part]

            pysics_part  = (u*Field_dx + v*Field_dy)*self.Dt
            Field_Dt     = Field_dt + pysics_part*self.reduce_Field_coef
            return [Field, Field_Dt, pysics_part] #(B,12,z,y,x)
