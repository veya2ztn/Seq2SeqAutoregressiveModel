#from tkinter.messagebox import NO
import numpy as np
import torch,os,io,socket
from torchvision import datasets, transforms
hostname = socket.gethostname()
if hostname not in ['SH-IDC1-10-140-0-184','SH-IDC1-10-140-0-185'] and '54' not in hostname and '52' not in hostname:
    from petrel_client.client import Client
    import petrel_client

from functools import lru_cache
import traceback
from tqdm import tqdm
import pandas as pd
from utils.timefeatures import time_features
import os
import h5py
from utils.tools import get_center_around_indexes,get_center_around_indexes_3D

def load_test_dataset_in_memory(years=[2018], root='cluster3:s3://era5npy',crop_coord=None,channel_last=True,vnames=[]):
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
    data = torch.empty(len(file_list),720,1440,20,pin_memory=True)
    #print(data.shape)
    for idx,(year, hour) in tqdm(enumerate(file_list)):
        arrays = []
        for name in vnames:
            url = f"{root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            if "s3://" in url:
                if client is None:client = Client(conf_path="~/petreloss.conf")
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
        if channel_last:arrays = arrays.transpose(1,2,0)
        data[idx]= torch.from_numpy(arrays)
    #data= torch.from_numpy(np.stack(data))

    return data

def getFalse(x):return False

def identity(x):
    return x

def do_batch_normlize(batch,mean,std):
    torchQ = isinstance(batch,torch.Tensor)
    if torchQ:
        mean=torch.Tensor(mean).to(batch.device)
        std =torch.Tensor(std).to(batch.device)
    
    if isinstance(batch,list):
        return [do_batch_normlize(x,mean,std) for x in batch]
    else:
        return (batch-mean)/(std+1e-10)

def inv_batch_normlize(batch,mean,std):
    torchQ = isinstance(batch,torch.Tensor)
    if torchQ:
        mean=torch.Tensor(mean).to(batch.device)
        std=torch.Tensor(std).to(batch.device)
    if isinstance(batch,list):
        return [inv_batch_normlize(x,mean,std) for x in batch]
    else:
        return batch*(std+1e-10)+mean


def read_npy_from_ceph(client, url, Ashape=(720,1440)):
    try:
        array_ceph = client.get(url)
        array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
    except:
        os.system(f"echo '{url}' > fail_data_path.txt")
    return array_ceph

def read_npy_from_buffer(path):
    buf = bytearray(os.path.getsize(path))
    with open(path, 'rb') as f:
        f.readinto(buf)
    return np.frombuffer(buf, dtype=np.half).reshape(720,1440)


class BaseDataset:
    client = None
    time_intervel=1
    def do_normlize_data(self, batch):
        return batch
    def inv_normlize_data(self,batch):
        return batch
    def set_time_reverse(self,time_reverse_flag):
        #if   time_reverse_flag == 'only_forward':
        #    self.do_time_reverse = lambda x:False
        #    print("we only using forward sequence, i.e. from t1, t2, ..., to tn")
        #elif time_reverse_flag == 'only_backward':
        #    self.do_time_reverse = lambda x:self.volicity_idx
        #    print("we only using backward sequence, i.e. from tn, tn-1, ..., to t1")
        #elif time_reverse_flag == 'random_forward_backward':
        #    self.do_time_reverse = lambda x:self.volicity_idx if np.random.random() > 0 else False
        #    print("we randomly(50%/50%) use forward/backward sequence")
        #else:
        #    raise NotImplementedError
        assert time_reverse_flag == 'only_forward'
        self.do_time_reverse = getFalse
    def do_time_reverse(self,idx):
        return False
    def __getitem__(self,idx):
        reversed_part = self.do_time_reverse(idx)
        time_step_list= [idx+i*self.time_intervel for i in range(self.time_step)]
        if reversed_part:time_step_list = time_step_list[::-1]
        batch = [self.get_item(i,reversed_part) for i in time_step_list]
        self.error_path = []
        return batch if not self.with_idx else (idx,batch)

        try:
            reversed_part = self.do_time_reverse(idx)
            time_step_list= [idx+i*self.time_intervel for i in range(self.time_step)]
            if reversed_part:time_step_list = time_step_list[::-1]
            batch = [self.get_item(i,reversed_part) for i in time_step_list]
            self.error_path = []
            return batch if not self.with_idx else (idx,batch)
        except:
            self.error_path.append(idx)
            if len(self.error_path) < 10:
                next_idx = np.random.randint(0,len(self))
                return self.__getitem__(next_idx)
            else:
                print(self.error_path)
                traceback.print_exc()
                raise NotImplementedError("too many error happened, check the errer path")
    

class ERA5BaseDataset(BaseDataset):
    full_vnames = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
        '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
        '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
        '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
        '50h_geopotential',
        'total_column_water_vapour',
    ]
    full_physics_index = [5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13]
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
    'all': range(1979, 2022)}
    
    
class ERA5CephDataset(ERA5BaseDataset):
    default_root = 'cluster3:s3://era5npy'
    default_time_step={'pretrain':2,'fintune':3}
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=None,
                class_name='ERA5Dataset', root= None,random_time_step=False,dataset_flag=False, ispretrain=True, crop_coord=None,time_step=None,with_idx=False,**kargs):
        
        if root is not None:
            self.root = root 
        else:
            self.root = root =self.default_root
        self.years = self.Years[split] if years is None else years
        self.crop_coord = crop_coord
        self.file_list = self.init_file_list()
        self.mode = mode
        self.channel_last = channel_last
        self.with_idx = with_idx
        self.error_path = []
        if self.mode   == 'pretrain':
            self.time_step = 2
        elif self.mode == 'finetune':
            self.time_step = 3
        elif self.mode == 'free5':
            self.time_step = 5
        if time_step is not None:self.time_step=time_step
        self.name  = f"{self.time_step}-step-task"
        self.random_time_step=random_time_step
        self.vnames= [self.full_vnames[t] for t in self.full_physics_index] if dataset_flag else self.full_vnames
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

    def load_numpy_from_url(self,url):
        if "s3://" in url:
            if self.client is None:self.client=Client(conf_path="~/petreloss.conf")
            array = read_npy_from_ceph(self.client, url)
        else:
            array = read_npy_from_buffer(url)
        return array

    @lru_cache(maxsize=32)
    def get_item(self, idx,reversed_part=False):
        year, hour = self.file_list[idx]
        arrays = []
        for name in self.vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            array = self.load_numpy_from_url(url)
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
            print(url)


class ERA5CephSmallDataset(ERA5CephDataset):
    clim_path = "datasets/era5G32x64_set/time_means.npy"
    dataset_path = {
            'train': "./datasets/era5G32x64_set/train_data.npy",
            'valid': "./datasets/era5G32x64_set/valid_data.npy",
            'test':  "./datasets/era5G32x64_set/test_data.npy"
        }
    img_shape = (32,64)
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,
                class_name='ERA5CephSmallDataset', ispretrain=True,
                crop_coord=None,root=None,
                dataset_tensor=None,record_load_tensor=None,time_step=None,with_idx=False,random_time_step=False,dataset_flag=False,
                time_reverse_flag='only_forward',time_intervel=1,use_time_stamp=False,**kargs):
        self.crop_coord   = crop_coord
        self.mode      = mode
        self.split=split
        if dataset_tensor is None:
            self.data,self.record_load_tensor = self.create_offline_dataset_templete(split)
        else:
            self.data,self.record_load_tensor = dataset_tensor,record_load_tensor
        self.clim_tensor = np.load(self.clim_path)
        
        ## [TODO] its better to offline below time stamp data
        if split == 'train':
            datatimelist  = np.arange(np.datetime64("1979-01-01"), np.datetime64("2016-01-01"), np.timedelta64(6, "h"))
        elif split == 'valid':
            datatimelist  = np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"), np.timedelta64(6, "h"))
        elif split == 'test':
            datatimelist  = np.arange(np.datetime64("2018-01-01"), np.datetime64("2022-01-01"), np.timedelta64(6, "h"))
        self.timestamp = time_features(pd.to_datetime(datatimelist)).transpose(1, 0)


        self.channel_pick = self.full_physics_index if dataset_flag=='physics' else list(range(20)) 
        
        self.vnames       = [self.full_vnames[t] for t in self.channel_pick]
        self.unit_list    = [self.mean_std_ERA5_20[name]['std'] for name in self.vnames]
        self.clim_tensor  = self.clim_tensor[:,self.channel_pick]
        self.channel_last = channel_last
        self.with_idx     = with_idx
        self.time_intervel= time_intervel
        self.error_path   = []
        self.random_time_step = random_time_step
        self.use_time_stamp = use_time_stamp
        if self.random_time_step:
            print("we are going use random step mode. in this mode, data sequence will randomly set 2-6 ")
        self.time_step = self.default_time_step[self.mode] if time_step is None else time_step

        self.time_reverse_flag = time_reverse_flag
        self.set_time_reverse(time_reverse_flag)
    
    @staticmethod
    def create_offline_dataset_templete(split='test',years=None, root=None,**kargs):
        print(f"in this dataset:{ERA5CephSmallDataset.__name__}, years/root args is disabled")
        data = torch.Tensor(np.load(ERA5CephSmallDataset.dataset_path[split]))
        record_load_tensor = torch.ones(len(data))
        return data,record_load_tensor

    def __len__(self):
        return len(self.data) - self.time_step*self.time_intervel + 1

    def get_item(self, idx,reversed_part=False):
        arrays = self.data[idx]
        if reversed_part:
            arrays = arrays.clone()#it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        arrays = arrays[self.channel_pick]
        if self.crop_coord is not None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            if isinstance(arrays,np.ndarray):
                arrays = arrays.transpose(1,2,0)
            else:
                arrays = arrays.permute(1,2,0)
        if self.use_time_stamp:
            return arrays, self.timestamp[idx]
        else:
            return arrays

class ERA5CephSmallPatchDataset(ERA5CephSmallDataset):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        self.cross_sample = kargs.get('cross_sample', True) and (self.split == 'train')
        self.patch_range = patch_range = kargs.get('patch_range', 5)
        self.center_index,self.around_index=get_center_around_indexes(self.patch_range,self.img_shape)
        self.channel_last = False
        
    def get_item(self, idx,reversed_part=False):
        arrays = self.data[idx]
        if reversed_part:
            arrays = arrays.clone()#it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        arrays = arrays[self.channel_pick]
        if self.cross_sample:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range//2)*2) 
            center_w = np.random.randint(self.img_shape[-1])
            patch_idx_h,patch_idx_w = self.around_index[center_h,center_w]
            arrays = arrays[..., patch_idx_h, patch_idx_w]

        # if self.crop_coord is not None:
        #     l, r, u, b = self.crop_coord
        #     arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            if isinstance(arrays,np.ndarray):
                arrays = arrays.transpose(1,2,0)
            else:
                arrays = arrays.permute(1,2,0)
        if self.use_time_stamp:
            return arrays, self.timestamp[idx]
        else:
            return arrays


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


class SpeedTestDataset(BaseDataset):
    def __init__(self, h, w , split="train", mode='pretrain', check_data=True,**kargs):
        self.mode = mode
        self.input_h = h
        self.input_w = w
        if self.mode   == 'pretrain':self.time_step = 2
        elif self.mode == 'finetune':self.time_step = 3
        elif self.mode == 'free5':self.time_step = 5
        pass

    def __len__(self):
        return 10000

    def __getitem__(self, ind):
        batch = [np.random.randn(self.input_h, self.input_w,20) for i in range(self.time_step)]
        self.error_path = []
        return batch


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


class WeathBench(BaseDataset):
    time_unit=1
    img_shape=(32,64)
    years_split={'train':range(1979, 2016),
           'valid':range(2016, 2018),
           'test':range(2018,2019),
           'ftest':range(1979,1980),
            'all': range(1979, 2022),
            'debug':range(1979,1980)}
    single_vnames = ["2m_temperature",
                      "10m_u_component_of_wind",
                      "10m_v_component_of_wind",
                      "total_cloud_cover",
                      "total_precipitation",
                      "toa_incident_solar_radiation"]
    level_vnames= []
    for physics_name in ["geopotential", "temperature",
                         "specific_humidity","relative_humidity",
                         "u_component_of_wind","v_component_of_wind",
                         "vorticity","potential_vorticity"]:
        for pressure_level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
            level_vnames.append(f"{pressure_level}hPa_{physics_name}")
    all_vnames = single_vnames + level_vnames
    cons_vnames= ["lsm", "slt", "orography"]
    default_root='datasets/weatherbench'
    constant_channel_pick = []
    one_single_data_shape = [110,32,64]
    reduce_Field_coef = np.array([1])
    datatimelist_pool={'train':np.arange(np.datetime64("1979-01-02"), np.datetime64("2016-01-01"), np.timedelta64(1, "h")),
                       'valid':np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"), np.timedelta64(1, "h")),
                        'test':np.arange(np.datetime64("2018-01-01"), np.datetime64("2019-01-01"), np.timedelta64(1, "h")),
                        'ftest':np.arange(np.datetime64("1979-01-02"), np.datetime64("1980-01-01"), np.timedelta64(1, "h")),
                        }
    use_offline_data =False
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,
                 root=None, time_step=2,
                 with_idx=False,
                 years=None,dataset_tensor=None,record_load_tensor=None,
                 dataset_flag='normal',time_reverse_flag='only_forward',time_intervel=1,use_time_stamp=False,
                 **kargs):
        if years is None:
            years = self.years_split[split]
        else:
            print(f"you are using flag={split}, the default range is {list(self.years_split[split])}")
            print(f"noice you assign your own year list {list(years)}")

        self.root             = self.default_root if root is None else root
        print(f"use dataset in {self.root}")
        self.split            = split
        self.single_data_path_list = self.init_file_list(years) # [[year,idx],[year,idx],.....]
        self.use_time_stamp = use_time_stamp
        # in memory dataset, if we activate the in memory procedure, please create shared tensor 
        # via Tensor.share_memory_() before call this module and pass the create tensor as args of this module
        # the self.dataset_tensor should be same shape as the otensor, for example (10000, 110, 32, 64)
        # the self.record_load_tensor is used to record which row is record.
        # This implement will automatively offline otensor after load it once.
        assert ((record_load_tensor is not None) and (dataset_tensor is not None)) or \
            ((record_load_tensor is    None) and (dataset_tensor is   None))
        
        if dataset_tensor is None:
            self.dataset_tensor,self.record_load_tensor = self.create_offline_dataset_templete(split,
            years=years, root=self.root, do_in_class=True,use_offline_data=self.use_offline_data,dataset_flag=dataset_flag)
        else:
            self.dataset_tensor,self.record_load_tensor = dataset_tensor,record_load_tensor

        self.dataset_flag     = dataset_flag
        self.clim_tensor      = [0]
        self.mean_std         = self.load_numpy_from_url(os.path.join(self.root,"mean_std.npy"))
        self.constants        = self.load_numpy_from_url(os.path.join(self.root,"constants.npy"))
        self.mode             = mode
        self.channel_last     = channel_last
        self.with_idx         = with_idx
        self.error_path       = []
        self.time_step        = time_step
        self.dataset_flag     = dataset_flag
        self.time_intervel    = time_intervel
        self.config_pool= config_pool= self.config_pool_initial()

        if dataset_flag in config_pool:
            self.channel_choice, self.normalize_type, mean_std, self.do_normlize_data , self.inv_normlize_data = config_pool[dataset_flag]
            self.mean,self.std = mean_std
            self.unit_list = self.std
            
        else:
            raise NotImplementedError
        
        self.volicity_idx   = ([1,2]+list(range(6+13*4,6+13*5))+list(range(6+13*5,6+13*6)))# we always assume the raw data is (P,y,x)
        self.time_reverse_flag = time_reverse_flag
        self.set_time_reverse(time_reverse_flag)
        self.vnames= [self.all_vnames[i] for i in self.channel_choice]
        self.vnames= self.vnames + [self.cons_vnames[i] for i in self.constant_channel_pick]
        self.timestamp = time_features(pd.to_datetime(self.datatimelist_pool[split])).transpose(1, 0) 
    @staticmethod
    def init_file_list(years):
        file_list = []
        for year in years:
            if year == 1979: # 1979年数据只有8753个，缺少第一天前7小时数据，所以这里我们从第二天开始算起
                for hour in range(17, 8753, 1):
                    file_list.append([year, hour])
            else:
                if year % 4 == 0:
                    max_item = 8784
                else:
                    max_item = 8760
                for hour in range(0, max_item, 1):
                    file_list.append([year, hour])
        return file_list

    @staticmethod
    def create_offline_dataset_templete(split='test',years=None, root=None, do_in_class=False,**kargs):
        if do_in_class:return None, None
        if years is None:
            years = WeathBench.years_split[split]
        else:
            print(f"you are using flag={split}, the default range is {list(WeathBench.years_split[split])}")
            print(f"noice you assign your own year list {list(years)}")

        root  = WeathBench.default_root if root is None else root
        batches = len(WeathBench.init_file_list(years))
        return torch.empty(batches,*WeathBench.one_single_data_shape),torch.zeros(batches)

    def config_pool_initial(self):
        config_pool={

            '2D110N': (list(range(110))  ,'gauss_norm'   , self.mean_std.reshape(2,110,1,1)      , identity, identity ),
            '2D110U': (list(range(110))  ,'unit_norm'    , self.mean_std.reshape(2,110,1,1)      , identity, identity ),
            '2D104N': (list(range(6,110)),'gauss_norm'   , self.mean_std[:,6:].reshape(2,104,1,1), identity, identity ),
            '2D104U': (list(range(6,110)),'unit_norm'    , self.mean_std[:,6:].reshape(2,104,1,1), identity, identity ),
            '3D104N': (list(range(6,110)),'gauss_norm_3D', self.mean_std[:,6:].reshape(2,8,13,1,1,1).mean(2), identity, identity ),
            '3D104U': (list(range(6,110)),'unit_norm_3D' , self.mean_std[:,6:].reshape(2,8,13,1,1,1).mean(2), identity, identity ),
        }

        # mean, std = self.mean_std.reshape(2,110,1,1)
        # config_pool['2D110O'] =(list(range(110))  ,'none', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))

        # mean, std = self.mean_std[:,6:].reshape(2,104,1,1)
        # config_pool['2D104O'] =(list(range(6,110)),'none', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))

        # mean, std = self.mean_std[:,6:].reshape(2,8,13,1,1,1).mean(2)
        # config_pool['3D104O'] =(list(range(6,110))  ,'3D', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))
        
        return config_pool



    def __len__(self):
        return len(self.single_data_path_list) - self.time_step*self.time_intervel + 1


    def get_item(self,idx,reversed_part=False):
        year, hour = self.single_data_path_list[idx]
        url = f"{self.root}/{year}/{year}-{hour:04d}.npy"
        odata= np.load(url)
        if reversed_part:odata[reversed_part] = -odata[reversed_part]
        data = odata[self.channel_choice]
        if '3D' in self.normalize_type:
            shape= data.shape
            data = data.reshape(8,13,*shape[-2:])
        if 'gauss_norm' in self.normalize_type:
            return (data - self.mean)/self.std
        elif 'unit_norm' in self.normalize_type:
            return data/self.std
        else:
            return data


class WeathBench71(WeathBench):
    default_root='datasets/weatherbench'
    _component_list= ([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  1]+ # u component of wind and the 10m u wind
                    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  2]+ # v component of wind and the 10m v wind
                    [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0]+ # Temperature and the 2m_temperature
                    [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18]+ # Geopotential and the last one is ground Geopotential, should be replace later
                    [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 57] # Realitve humidity and the Realitve humidity at groud, should be modified by total precipitaiton later
                    )
    volicity_idx = ([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  1]+ # u component of wind and the 10m u wind
                 [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  2] # v component of wind and the 10m v wind
                )
    use_offline_data = False            
    def config_pool_initial(self):

        _list = self._component_list
        vector_scalar_mean = self.mean_std.copy()
        vector_scalar_mean[:,self.volicity_idx] = 0
        config_pool={
            '2D70V': (_list ,'gauss_norm'   , vector_scalar_mean[:,_list].reshape(2,70,1,1), identity, identity ),
            '2D70N': (_list ,'gauss_norm'   , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            '2D70U': (_list ,'unit_norm'    , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            '3D70N': (_list ,'gauss_norm_3D', self.mean_std[:,_list].reshape(2,5,14,1,1,1).mean(2), identity, identity ),
            '3D70U': (_list ,'unit_norm_3D' , self.mean_std[:,_list].reshape(2,5,14,1,1,1).mean(2), identity, identity ),
        }


        # mean, std = self.mean_std[:,_list].reshape(2,70,1,1)
        # config_pool['2D70O'] =(_list,'none', (0,1) , lambda x:do_batch_normlize(x,mean,std), lambda x:inv_batch_normlize(x,mean,std))
        # mean, std = self.mean_std[:,_list].reshape(2,5,14,1,1,1).mean(2)
        # config_pool['3D70O'] =(_list  ,'3D', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))
        return config_pool

    def load_numpy_from_url(self,url):#the saved numpy is not buffer, so use normal reading
        if "s3://" in url:
            if self.client is None:self.client=Client(conf_path="~/petreloss.conf")
            with io.BytesIO(self.client.get(url)) as f:
                array = np.load(f)
        else:
            array = np.load(url)
        return array

    def load_otensor(self,idx):
        if (self.record_load_tensor is None) or (not self.record_load_tensor[idx]):
            year, hour = self.single_data_path_list[idx]
            url = f"{self.root}/{year}/{year}-{hour:04d}.npy"
            odata= self.load_numpy_from_url(url)
            if self.record_load_tensor is not None: 
                self.record_load_tensor[idx] = 1
                self.dataset_tensor[idx] = torch.Tensor(odata)
        if self.record_load_tensor is not None:
            return self.dataset_tensor[idx].clone()
        else:
            return odata
    
    def get_item(self,idx,reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        if self.use_offline_data:
            data =  self.dataset_tensor[idx]
        else:
            odata=self.load_otensor(idx)
            if reversed_part:
                odata = odata.clone() if isinstance(odata,torch.Tensor) else odata.copy()
                odata[reversed_part] = -odata[reversed_part]
            data = odata[self.channel_choice]
            eg = torch if isinstance(data,torch.Tensor) else np
            if '3D' in self.normalize_type:
                # 3D should modify carefully
                data[14*4-1] = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
                total_precipitaiton = odata[4]
                newdata = data[14*5-1].clone() if isinstance(data,torch.Tensor) else data[14*5-1].copy()
                newdata[total_precipitaiton>0] = 100
                newdata[data[14*5-1]>100]=data[14*5-1][data[14*5-1]>100]
                data[14*5-1] = newdata
                shape= data.shape
                data = data.reshape(5,14,*shape[-2:])
            else:
                data[14*4-1]    = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
                total_precipitaiton = odata[4]
                data[14*5-1]    = total_precipitaiton
            
            if 'gauss_norm' in self.normalize_type:
                data=  (data - self.mean)/self.std
            elif 'unit_norm' in self.normalize_type:
                data = data/self.std

        if self.use_time_stamp:
            return data, self.timestamp[idx]
        else:
            return data

class WeathBench71_H5(WeathBench71):

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

class WeathBench706(WeathBench71):

    def __init__(self,**kargs):
        super().__init__(**kargs)
        self.single_data_path_list = self.single_data_path_list[::6]



class WeathBench716(WeathBench71):
    default_root='datasets/weatherbench_6hour'
    use_offline_data = False
    def init_file_list(self,years):
        return np.load(os.path.join(self.root,f"{self.split}.npy"))
    def load_otensor(self,idx):
        return self.single_data_path_list[idx]
    def config_pool_initial(self):
        _list = self._component_list
        self.constant_channel_pick = [0]
        mean_std = self.mean_std[:,_list].reshape(2,70,1,1)
        mean, std= mean_std #(70,1,1)
        mean = np.concatenate([mean,np.zeros(1).reshape(1,1,1)])
        std  = np.concatenate([ std, np.ones(1).reshape(1,1,1)])
        mean_std = np.stack([mean,std])
        
        config_pool={
            '2D71N': (_list ,'gauss_norm'   , mean_std, identity, identity ),
            '2D71U': (_list ,'unit_norm'    , mean_std, identity, identity ),
        }
        return config_pool

    def get_item(self, idx,reversed_part=False):
        arrays = self.load_otensor(idx)
        if reversed_part:
            arrays = arrays.copy()#it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        constant= self.constants[self.constant_channel_pick]
        arrays  = np.concatenate([arrays[self.channel_choice],constant])
        
        if self.channel_last:
            if isinstance(arrays,np.ndarray):
                arrays = arrays.transpose(1,2,0)
            else:
                arrays = arrays.permute(1,2,0)
        return arrays

class WeathBench7066(WeathBench71):
    default_root='datasets/weatherbench_6hour'
    use_offline_data = False
    time_unit=6
    datatimelist_pool={'train':np.arange(np.datetime64("1979-01-02"), np.datetime64("2017-01-01"), np.timedelta64(6, "h")),
                       'valid':np.arange(np.datetime64("2017-01-01"), np.datetime64("2018-01-01"), np.timedelta64(6, "h")),
                        'test':np.arange(np.datetime64("2018-01-01"), np.datetime64("2019-01-01"), np.timedelta64(6, "h")),
                        'ftest':np.arange(np.datetime64("1979-01-02"), np.datetime64("1980-01-01"), np.timedelta64(6, "h"))}
    def __init__(self,**kargs):
        use_offline_data = kargs.get('use_offline_data',0) 
        if use_offline_data ==1:
            print("use offline data mode <1>: only train use offline data")
            self.use_offline_data = (kargs.get('split')=='train')
        if use_offline_data ==2:
            print("use offline data mode <2>: train/valid/test use offline data")
            self.use_offline_data = 1

        super().__init__(**kargs)

    def __len__(self):
        return len(self.dataset_tensor) - self.time_step*self.time_intervel + 1
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:root = WeathBench7066.default_root
        if use_offline_data:
            dataset_flag = kargs.get('dataset_flag')
            data_name = f"{split}_{dataset_flag}.npy"
        else:
            data_name = f"{split}.npy"
        numpy_path = os.path.join(root,data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor   = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor

    def load_otensor(self,idx):
        data = self.dataset_tensor[idx]
        data = data*self.mean_std[1].reshape(110,1,1) + self.mean_std[0].reshape(110,1,1)
        return data

    def config_pool_initial(self):
        _list = self._component_list
        vector_scalar_mean = self.mean_std.copy()
        vector_scalar_mean[:,self.volicity_idx] = 0
        config_pool={
            '2D70V': (_list ,'gauss_norm'   , vector_scalar_mean[:,_list].reshape(2,70,1,1), identity, identity ),
            '2D70N': (_list ,'gauss_norm'   , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            '2D70U': (_list ,'unit_norm'    , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            '3D70N': (_list ,'gauss_norm_3D', self.mean_std[:,_list].reshape(2,5,14,1,1), identity, identity ),
            #'3D70U': (_list ,'unit_norm_3D' , self.mean_std[:,_list].reshape(2,5,14,1,1,1).mean(2), identity, identity ),
        }
        
        # mean, std = self.mean_std[:,_list].reshape(2,70,1,1)
        # config_pool['2D70O'] =(_list,'none', (0,1) , lambda x:do_batch_normlize(x,mean,std), lambda x:inv_batch_normlize(x,mean,std))
        # mean, std = self.mean_std[:,_list].reshape(2,5,14,1,1)
        # config_pool['3D70O'] =(_list  ,'3D', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))
        return config_pool
    
class WeathBench7066PatchDataset(WeathBench7066):
    def __init__(self,**kargs):
        self.use_offline_data = kargs.get('use_offline_data',0) and kargs.get('split')=='train'
        super().__init__(**kargs)
        self.cross_sample     = kargs.get('cross_sample', True) and (self.split == 'train')
        
        self.patch_range      = patch_range = kargs.get('patch_range')
        #self.img_shape        = kargs.get('img_size',WeathBench7066PatchDataset.img_shape)
        #if isinstance(self.img_shape,str):self.img_shape=tuple([int(p) for p in self.img_shape.split(',')])
        self.img_shape       = WeathBench7066PatchDataset.img_shape
        #print(self.img_shape)
        

        if '3D' in self.normalize_type:
            #self.img_shape        = kargs.get('img_size',WeathBench7066PatchDataset.img_shape)
            #if isinstance(self.img_shape,str):self.img_shape=tuple([int(p) for p in self.img_shape.split(',')])
            self.img_shape = (14,32,64)
            self.center_index,self.around_index = get_center_around_indexes_3D(self.patch_range,self.img_shape)
        else:
            self.center_index,self.around_index = get_center_around_indexes(self.patch_range,self.img_shape)
        print(f"notice we will use around_index{self.around_index.shape} to patch data")
        self.channel_last                   = False
        self.random = kargs.get('random_dataset', False)
        self.use_position_idx = kargs.get('use_position_idx', False)

    def get_item(self,idx,location=None,reversed_part=False):
        if self.use_offline_data:
            data =  self.dataset_tensor[idx]
        else:
            odata=self.load_otensor(idx)
            
            if reversed_part:
                odata = odata.clone() if isinstance(odata,torch.Tensor) else odata.copy()
                odata[reversed_part] = -odata[reversed_part]
            data = odata[self.channel_choice]
            
            eg = torch if isinstance(data,torch.Tensor) else np
            if '3D' in self.normalize_type:
                # 3D should modify carefully
                data[14*4-1] = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
                total_precipitaiton = odata[4]
                newdata = data[14*5-1].clone() if isinstance(data,torch.Tensor) else data[14*5-1].copy()
                newdata[total_precipitaiton>0] = 100
                newdata[data[14*5-1]>100]=data[14*5-1][data[14*5-1]>100]
                data[14*5-1] = newdata
                shape= data.shape
                data = data.reshape(5,14,*shape[-2:])
            else:
                data[14*4-1]    = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
                total_precipitaiton = odata[4]
                data[14*5-1]    = total_precipitaiton
            
            if 'gauss_norm' in self.normalize_type:
                data=  (data - self.mean)/self.std
            elif 'unit_norm' in self.normalize_type:
                data = data/self.std

        if location is not None:
            if '3D' in self.normalize_type:
                assert patch_idx_z is not None
                patch_idx_z, patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_z, patch_idx_h, patch_idx_w]
            else:
                patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_h, patch_idx_w]
        else:
            location = -1
        out = [data]
        if self.use_time_stamp:
            out.append(self.timestamp[idx])
        if self.use_position_idx:
            out.append(location)
        if len(out)==1:out=out[0]
        return out

    def __getitem__(self,idx):
        if self.random:
            idx = np.random.randint(self.__len__())
        reversed_part = self.do_time_reverse(idx)
        time_step_list= [idx+i*self.time_intervel for i in range(self.time_step)]
        if reversed_part:time_step_list = time_step_list[::-1]
        patch_idx_h = patch_idx_w = patch_idx_z = None
        location = None
        if self.cross_sample:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range//2)*2) 
            center_w = np.random.randint(self.img_shape[-1])
            if '3D' in self.normalize_type:
                center_z = np.random.randint(self.img_shape[-3] - (self.patch_range//2)*2) 
                location      = self.around_index[center_z,center_h, center_w]
                #location = self.center_index[:,center_z, center_h, center_w]
            else:
                location = self.around_index[center_h, center_w]
                #location= self.center_index[:, center_h, center_w]
        batch = [self.get_item(i,location=location,
                      reversed_part=reversed_part) 
                     for i in time_step_list]
        self.error_path = []
        return batch if not self.with_idx else (idx,batch)



if __name__ == "__main__":
    import sys
    import time
    from tqdm import tqdm
    from multiprocessing import Pool
    from petrel_client.client import Client
    import petrel_client
    dataset = ERA5CephDataset(split='train')
    print(len(dataset[0]))
  
    def load_data_range(dataset, start, end):
        for i in tqdm(range(start, end)):
            _ = dataset[i]

    def parallel_load_data(dataset):
        total_len = len(dataset)
        processor = 10
        range_list = np.linspace(0, total_len, processor+1)
        range_list = [int(a) for a in range_list]
        print(range_list)
        res = []
        p = Pool(processor)
        for i in range(processor):
            res.append(p.apply_async(load_data_range, args=(
                dataset, range_list[i], range_list[i+1])))
            print(str(i) + ' processor started !')
        p.close()
        p.join()

    def check_range(dataset, start, end):
        fail_list = []
        for i in tqdm(range(start, end)):
            if i > len(dataset):
                continue
            now_time = time.time()
            try:
                _ = dataset[i]
            except:
                print(f"fail at {i}")
            fail_list.append(i)
            cost = time.time() - now_time
            #print(cost)
        return fail_list
    dataset = ERA5CephDataset(split='train')
    print(ERA5CephDataset.__name__)
    print(dataset[0])
    
    print(len(dataset[0]))
    print(len(dataset[1]))


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
