#from tkinter.messagebox import NO
import numpy as np
import torch,os
from torchvision import datasets, transforms
try:
    from petrel_client.client import Client
    import petrel_client
except:
    print("can not input petrel client, pass")
    pass
from functools import lru_cache
import traceback
from tqdm import tqdm
vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
]
physice_vnames=[vnames[t] for t in [5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13]]
vnames_short = [
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
Years = range(1979, 2022)
# Years4Train = range(1979, 2016)
Years4Train = range(1979, 1987)
Years4Valid = range(2016, 2018)
Years4Test = range(2018, 2020)
Years = {
    'train': range(1979, 2016),
    'valid': range(2016, 2018),
    'test': range(2018, 2022),
    'all': range(1979, 2022)

}

Shape = (720, 1440)
import os
import h5py

smalldataset_path={
    'train': "./datasets/era5G32x64_set/train_data.npy",
    'valid': "./datasets/era5G32x64_set/valid_data.npy",
    'test':  "./datasets/era5G32x64_set/test_data.npy"
}

def load_small_dataset_in_memory(split):
    return torch.Tensor(np.load(smalldataset_path[split]))
def load_test_dataset_in_memory(years=[2018], root='cluster3:s3://era5npy',crop_coord=None,channel_last=True,vnames=vnames):
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
    data = torch.empty(len(file_list),720,1440,20)
    #print(data.shape)
    for idx,(year, hour) in tqdm(enumerate(file_list)):
        arrays = []
        for name in vnames:
            url = f"{root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            if "s3://" in url:
                if client is None:client = Client(conf_path="~/petreloss.conf")
                array = ERA5CephDataset.read_npy_from_ceph(client, url)
            else:
                array = ERA5CephDataset.read_npy_from_buffer(url)
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

class BaseDataset:
    def do_normlize_data(self, batch):
        return batch
    def inv_normlize_data(self,batch):
        return batch
    def __getitem__(self,idx):
        try:
            batch = [self.get_item(idx+i) for i in range(self.time_step)]
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

class ERA5CephDataset(BaseDataset):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=None,
                class_name='ERA5Dataset', root='cluster3:s3://era5npy',random_time_step=False,enable_physics_dataset=False, ispretrain=True, crop_coord=None,time_step=None,with_idx=False,**kargs):
        self.client = Client(conf_path="~/petreloss.conf")
        self.root = root
        self.years = Years[split] if years is None else years
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
        self.vnames= physice_vnames if enable_physics_dataset else vnames
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

    @staticmethod
    def read_npy_from_ceph(client, url, Ashape=Shape):
        try:
            array_ceph = client.get(url)
            array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
        except:
            os.system(f"echo '{url}' > fail_data_path.txt")
        return array_ceph
    @staticmethod
    def read_npy_from_buffer(path):
        buf = bytearray(os.path.getsize(path))
        with open(path, 'rb') as f:
            f.readinto(buf)
        return np.frombuffer(buf, dtype=np.half).reshape(720,1440)


    @lru_cache(maxsize=32)
    def get_item(self, idx):
        year, hour = self.file_list[idx]
        arrays = []
        for name in self.vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            if "s3://" in url:
                array = self.read_npy_from_ceph(self.client, url)
            else:
                array = self.read_npy_from_buffer(url)
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
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,
                class_name='ERA5CephSmallDataset', ispretrain=True,
                crop_coord=None,
                dataset_tensor=None,
                time_step=None,with_idx=False,random_time_step=False,enable_physics_dataset=False):
        self.crop_coord   = crop_coord
        self.mode         = mode
        self.data         = load_small_dataset_in_memory(split) if dataset_tensor is None else dataset_tensor
        self.vnames= vnames
        smalldataset_clim_path = "datasets/era5G32x64_set/time_means.npy"
        self.clim_tensor = np.load(smalldataset_clim_path)
        if enable_physics_dataset:
            self.data = self.data[:,[5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13]]
            self.vnames= physice_vnames
            self.clim_tensor= self.clim_tensor[:,[5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13]]
        self.channel_last = channel_last
        self.with_idx     = with_idx
        self.error_path   = []
        self.random_time_step = random_time_step
        if self.random_time_step:
            print("we are going use random step mode. in this mode, data sequence will randomly set 2-6 ")
        if self.mode   == 'pretrain':self.time_step = 2
        elif self.mode == 'finetune':self.time_step = 3
        elif self.mode == 'free5':self.time_step = 5
        if time_step is not None:
            self.time_step = time_step
    def __len__(self):
        return len(self.data) - self.time_step + 1

    def get_item(self, idx):
        arrays = self.data[idx]
        if self.crop_coord is not None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            if isinstance(arrays,np.ndarray):
                arrays = arrays.transpose(1,2,0)
            else:
                arrays = arrays.permute(1,2,0)
        return arrays


class ERA5CephInMemoryDataset(ERA5CephDataset):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=[2018],dataset_tensor=None,
                class_name='ERA5Dataset', root='cluster3:s3://era5npy', ispretrain=True, crop_coord=None,time_step=None,
                with_idx=False,enable_physics_dataset=False,**kargs):
        if dataset_tensor is None:
            self.data=load_test_dataset_in_memory(years=years,root=root,crop_coord=crop_coord,channel_last=channel_last)
        else:
            self.data= dataset_tensor
        self.vnames= vnames
        if enable_physics_dataset:
            self.data = self.data[:,[5, 9,14 , 6,10,15 ,7,11,16 ,2, 8,13]]
            self.vnames= physice_vnames
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


class ERA5Tiny12_47_96(BaseDataset):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=[2018],dataset_tensor=None,
                class_name='ERA5Dataset', root='datasets/ERA5/h5_set', ispretrain=True, crop_coord=None,time_step=2,
                with_idx=False,enable_physics_dataset='normal',**kargs):
        if dataset_tensor is None:
            self.Field_dx= h5py.File(os.path.join(root,f"Field_dx_{split}.h5"), 'r')['data']
            self.Field_dy= h5py.File(os.path.join(root,f"Field_dy_{split}.h5"), 'r')['data']
            self.Field   = h5py.File(os.path.join(root,f"all_data_{split}.h5"), 'r')['data']
            self.Dt      = 6*3600
        else:
            raise NotImplementedError
        self.enable_physics_dataset = enable_physics_dataset
        self.vnames= [vnames[t] for t in [5, 9,14 , 6,10,15  ,2, 8,13,7,11,16]]
        self.mode  = mode
        self.channel_last = channel_last
        self.with_idx = with_idx
        self.error_path = []
        self.clim_tensor= np.load(os.path.join(root,f"time_means.npy"))[...,1:-1,:].reshape(4,3,47,96)

        # if mode != 'fourcast':
        #     self.time_step = 1
        #     print(f"in mode=[{mode}], we will force time_step={self.time_step}")
        # else:
        #     self.time_step = time_step
        self.time_step = time_step

        self.Field_channel_mean    = np.array([2.7122362e+00,9.4288319e-02,2.6919699e+02,2.2904861e+04]).reshape(4,1,1,1)
        self.Field_channel_std     = np.array([9.5676870e+00,7.1177821e+00,2.0126169e+01,2.2861252e+04]).reshape(4,1,1,1)
        self.Field_Dt_channel_mean =    np.array([  -0.02293313,-0.04692488  ,0.02711264   ,7.51324121]).reshape(4,1,1,1)
        self.Field_Dt_channel_std  =  np.array([  8.82677214 , 8.78834556  ,3.96441518   ,526.15269219]).reshape(4,1,1,1)

        if enable_physics_dataset == 'reduce':
            self.reduce_Field_coef    = np.array([-0.008569032217562018, -0.09391911216803356, -0.05143135231455811, -0.10192078794347732]).reshape(4,1,1,1)
            self.reduce_Field_Dt_channel_mean = np.array([2.16268301e-04 ,4.40459576e-03,-1.36775636e-03,-7.66760608e-01]).reshape(4,1,1,1)
            self.reduce_Field_Dt_channel_std  =  np.array([  3.26819142  ,4.04325207 , 1.99422196 ,235.31884021]).reshape(4,1,1,1)
        else:
            self.reduce_Field_coef = 1
        self.Field_mean     = None
        self.Field_std      = None
        self.Field_Dt_mean  = None
        self.Field_Dt_std   = None
    def __len__(self):
        return len(self.Field) - self.time_step + 1

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
            self.Field_Dt_mean = self.reduce_Field_Dt_channel_mean if self.enable_physics_dataset == 'reduce' else self.Field_Dt_channel_mean
            self.Field_Dt_std  = self.reduce_Field_Dt_channel_std if self.enable_physics_dataset == 'reduce' else self.Field_Dt_channel_std
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

    def get_item(self,idx):
        Field        = self.Field[idx][...,1:-1,:]#(P,3,47,96)
        NextField    = self.Field[idx+1][...,1:-1,:]#(P,3,47,96)
        u            = Field[0:1]
        v            = Field[1:2]
        T            = Field[2:3]
        p            = Field[3:4]
        Field_dt     = NextField - Field #(P,3,47,96)
        Field_dx     = self.Field_dx[idx]#(P,3,47,96)
        Field_dy     = self.Field_dy[idx]#(P,3,47,96)
        pysics_part  = (u*Field_dx + v*Field_dy)*self.Dt
        Field_Dt     = Field_dt + pysics_part*self.reduce_Field_coef
        return [Field, Field_Dt, pysics_part] #(B,12,z,y,x)

class ERA5Tiny12_47_96_Normal(ERA5Tiny12_47_96):
    def get_item(self,idx):
        Field        = self.Field[idx][...,1:-1,:]#(P,3,47,96)
        return Field

if __name__ == "__main__":
    import sys
    import time
    from tqdm import tqdm
    from multiprocessing import Pool

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
    raise
    print(len(dataset))
    dataset = ERA5CephDataset(split='valid')
    print(len(dataset))
    raise
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
