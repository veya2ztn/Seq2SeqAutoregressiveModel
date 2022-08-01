
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
vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',

]
vnames_short = [
    'u10', 'v10', 't2m', 'sp', 'msl',
    'u', 'v', 'z',
    't', 'u', 'v', 'z', 'r',
    't', 'u', 'v', 'z', 'r',
    'z',
    'tcwv'

]

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
class ERA5CephDataset(datasets.ImageFolder):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,years=None,
                class_name='ERA5Dataset', root='cluster3:s3://era5npy', ispretrain=True, crop_coord=None,time_step=None,with_idx=False,**kargs):
        self.client = Client(conf_path="~/petreloss.conf")
        self.root = root
        self.years = Years[split] if years is None else years
        self.crop_coord = crop_coord
        self.file_list = self.init_file_list()
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
        for name in vnames:
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
        for name in vnames:
            url = f"{self.root}/{name}/{year}/{name}-{year}-{hour:04d}.npy"
            print(url)

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
        else:
            raise NotImplementedError(f"mode={self.mode} is not supported")


smalldataset_path={
    'train': "./datasets/era5G32x64/train_data.npy",
    'valid': "./datasets/era5G32x64/valid_data.npy",
    'test':  "./datasets/era5G32x64/test_data.npy"
}
def load_small_dataset_in_memory(split):
    return torch.Tensor(np.load(smalldataset_path[split]))
class ERA5CephSmallDataset(ERA5CephDataset):
    def __init__(self, split="train", mode='pretrain', channel_last=True, check_data=True,
                class_name='ERA5CephSmallDataset', ispretrain=True, crop_coord=None,dataset_tensor=None,time_step=None,with_idx=False):
        self.crop_coord   = crop_coord
        self.mode         = mode
        self.data         = load_small_dataset_in_memory(split) if dataset_tensor is None else dataset_tensor
        self.channel_last = channel_last
        self.with_idx     = with_idx
        self.error_path   = []
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


class SpeedTestDataset(torch.utils.data.Dataset):
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
