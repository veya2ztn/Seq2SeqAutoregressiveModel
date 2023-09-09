from .base import BaseDataset, identity
import numpy as np 
import os
import torch
import pandas as pd
import copy
from utils.tools import get_sub_luna_point,get_sub_sun_point
from utils.timefeatures import time_features
from .normlizer import GauessNormlizer,UnitNormlizer,NoneNormlizer
class WeathBench(BaseDataset):
    time_unit  = 1
    img_shape  = (32,64)
    years_split={'train':range(1979, 2016),
                 'valid':range(2016, 2018),
                 'test':range(2018,2019),
                 'ftest':range(1979,1980),
                 'all': range(1979, 2022),
                 'debug':range(1979,1980)}
    single_vnames = [ "2m_temperature",
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
    threeD_channel = np.arange(6, 110).reshape(8,13)
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
    clim_tensor = [0]
    
    def __init__(self, split="train", shared_dataset_tensor_buffer=(None,None),
                 config = {}):
        self.root = config.get('root', self.default_root)
        self.years = config.get('years', None)
        if self.years is None:
            self.years=self.Years[split] 
        else:
            print(f"you are using flag={split}, the default range is {list(self.years_split[split])}")
            print(f"noice you assign your own year list {list(self.years)}")
        self.time_step         = config.get('time_step', 2)
        self.crop_coord        = config.get('crop_coord', None)
        self.channel_last      = config.get('channel_last', None)
        self.with_idx          = config.get('with_idx', False)
        self.dataset_flag      = config.get('dataset_flag', 'normal')
        self.use_time_stamp    = config.get('use_time_stamp', False)
        self.time_reverse_flag = config.get('time_reverse_flag', False)
        self.use_offline_data  = config.get('use_offline_data', False)
        self.time_intervel     = config.get('time_intervel', 1)


        print(f"use dataset in {self.root}")
        self.split            = split
        self.single_data_path_list = self.init_file_list(years) # [[year,idx],[year,idx],.....]

        # In-memory dataset, if we activate the in memory procedure, please create shared tensor 
        # via Tensor.share_memory_() before call this module and pass the create tensor as args of this module
        # the self.dataset_tensor should be same shape as the otensor, for example (10000, 110, 32, 64)
        # the self.record_load_tensor is used to record which row is record.
        # This implement will automatively offline otensor after load it once.

        ## no matter we use in-memory technology or not, we always initialized two tensor,
        ## one is the whole tensor dataset stored in (B, P, W, H) tensor 
        ## one is the loadQ tensor that judge whether the real data is loaded.
        assert ((shared_dataset_tensor_buffer[0] is None and shared_dataset_tensor_buffer[1] is None) or
                (self.use_offline_data and shared_dataset_tensor_buffer[0] is not None and shared_dataset_tensor_buffer[1] is not None))
        
        if shared_dataset_tensor_buffer[0] is None:
            self.dataset_tensor, self.record_load_tensor = self.create_offline_dataset_templete(split,
            years=self.years, root=self.root, do_in_class=True,
            use_offline_data=self.use_offline_data,
            dataset_flag=self.dataset_flag)
        else:
            self.dataset_tensor,self.record_load_tensor = shared_dataset_tensor_buffer

        self.mean_std         = self.load_numpy_from_url(os.path.join(self.root,"mean_std.npy")) #(2, P, W, H)
        self.constants        = self.load_numpy_from_url(os.path.join(self.root,"constants.npy"))
        self.config_pool      = config_pool= self.config_pool_initial()


        self.channel_choice, self.normlizer = self.config_pool[dataset_flag]
        self.mean,self.std  = mean_std
        self.unit_list      = self.std
        
        

        if self.time_reverse_flag:
            ### TODO:it is better use vname to select the volicity 
            self.volicity_idx = ([1,2]+list(range(6+13*4,6+13*5))+list(range(6+13*5,6+13*6)))# we always assume the raw data is (P,y,x)

        self.vnames= [self.all_vnames[i] for i in self.channel_choice]
        self.vnames= self.vnames + [self.cons_vnames[i] for i in self.constant_channel_pick]

        if self.use_time_stamp:
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
        if do_in_class:return None, None #<---why skip?
        if years is None:
            years = WeathBench.years_split[split]
        else:
            print(f"you are using flag={split}, the default range is {list(WeathBench.years_split[split])}")
            print(f"noice you assign your own year list {list(years)}")

        root  = WeathBench.default_root if root is None else root
        batches = len(WeathBench.init_file_list(years))
        return torch.empty(batches,*WeathBench.one_single_data_shape),torch.zeros(batches)

    def get_config_from_preset_pool(self, dataset_flag):
        #### this method should be optimized, otherwise it may cause timeconsume
        if   dataset_flag == '2D110N':return list(range(0,110)) , GauessNormlizer(self.mean_std.reshape(2, 110, 1, 1))
        elif dataset_flag == '2D110U':return list(range(0,110)) ,   UnitNormlizer(self.mean_std.reshape(2, 110, 1, 1))
        elif dataset_flag == '2D104N':return list(range(6,110)) , GauessNormlizer(self.mean_std[:,6:].reshape(2, 104, 1, 1))
        elif dataset_flag == '2D104U':return list(range(6,110)) ,   UnitNormlizer(self.mean_std[:,6:].reshape(2, 104, 1, 1))
        elif dataset_flag == '3D104N':return self.threeD_channel, GauessNormlizer(self.mean_std[:,6:].reshape(2,8,13, 1, 1).mean(2))
        elif dataset_flag == '3D104U':return self.threeD_channel,   UnitNormlizer(self.mean_std[:,6:].reshape(2,8,13, 1, 1).mean(2))

        ### below is for post - normlized data case
        # mean, std = self.mean_std.reshape(2,110,1,1)
        # config_pool['2D110O'] =(list(range(110))  ,'none', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))

        # mean, std = self.mean_std[:,6:].reshape(2,104,1,1)
        # config_pool['2D104O'] =(list(range(6,110)),'none', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))

        # mean, std = self.mean_std[:,6:].reshape(2,8,13,1,1,1).mean(2)
        # config_pool['3D104O'] =(list(range(6,110))  ,'3D', (0,1) , lambda x:do_batch_normlize(x,mean,std),lambda x:inv_batch_normlize(x,mean,std))


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

    def get_mesh_lon_lat(self,tH=32,tW=64):
        sH = 720
        sW = 1440
        steph = sH /float(tH)
        stepw = sW /float(tW)
        x     = np.arange(0, sW, stepw).astype('int')
        y     = np.linspace(0, sH-1, tH, dtype='int')
        x, y  = np.meshgrid(x, y)
        latitude   = np.linspace(90,-90,721)
        longitude  = np.linspace(0,360,1440)
        LaLotude = np.stack([latitude[y],longitude[x]])/180*np.pi
        LaLotudeVector = np.stack([np.cos(LaLotude[1])*np.cos(LaLotude[0]),np.cos(LaLotude[1])*np.sin(LaLotude[0]),np.sin(LaLotude[1])],2)
        return LaLotude,LaLotudeVector
 
class WeathBench71(WeathBench):
    default_root='datasets/weatherbench'
    _component_list= ([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  1]+ # u component of wind and the 10m u wind
                      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  2]+   # v component of wind and the 10m v wind
                      [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0]+   # Temperature and the 2m_temperature
                      [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18]+   # Geopotential and the last one is ground Geopotential, should be replace later
                      [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 57]    # Realitve humidity and the Realitve humidity at groud, should be modified by total precipitaiton later
                    )
    _component_list68= ([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  1]+ # u component of wind and the 10m u wind
                        [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  2]+ # v component of wind and the 10m v wind
                        [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0]+ # Temperature and the 2m_temperature
                        [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]+ # Geopotential 
                        [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57] # Realitve humidity 
                    )  
    _component_list55= ([58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  1]+ # u component of wind and the 10m u wind
                        [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  2]+ # v component of wind and the 10m v wind
                        [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0]+ # Temperature and the 2m_temperature
                        [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18] # Geopotential and the last one is ground Geopotential, should be replace later
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
            '2D68N': (self._component_list68 ,'gauss_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
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
    
    def generate_runtime_data(self,idx,reversed_part=False):
        odata=self.load_otensor(idx)
        if reversed_part:
            odata = odata.clone() if isinstance(odata,torch.Tensor) else odata.copy()
            odata[reversed_part] = -odata[reversed_part]
        data = odata[self.channel_choice]
        eg = torch if isinstance(data,torch.Tensor) else np
        if '3D70' in self.normalize_type:
            # 3D should modify carefully
            data[14*4-1] = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
            total_precipitaiton = odata[4]
            newdata = data[14*5-1].clone() if isinstance(data,torch.Tensor) else data[14*5-1].copy()
            newdata[total_precipitaiton>0] = 100
            newdata[data[14*5-1]>100]=data[14*5-1][data[14*5-1]>100]
            data[14*5-1] = newdata
            shape= data.shape
            data = data.reshape(5,14,*shape[-2:])
        elif '2D70' in self.normalize_type:
            data[14*4-1]    = eg.ones_like(data[14*4-1])*50 # modifiy the groud Geopotential, we use 5m height
            total_precipitaiton = odata[4]
            data[14*5-1]    = total_precipitaiton
        
        if 'gauss_norm' in self.normalize_type:
            data=  (data - self.mean)/self.std
        elif 'unit_norm' in self.normalize_type:
            data = data/self.std
        return data

    def get_item(self,idx,reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        if self.use_offline_data:
            data =  self.dataset_tensor[idx]
        else:
            data = self.generate_runtime_data(idx,reversed_part=reversed_part)
        
        if self.use_time_stamp:
            return data, self.timestamp[idx]
        else:
            return data


class WeathBench32x64(WeathBench71):
    default_root = 'datasets/weatherbench32x64'

class WeathBench32x64SPnorm(WeathBench32x64):
    space_time_mean = None
    space_time_std  = None

    def __init__(self, **kargs):
        use_offline_data = kargs.get('use_offline_data', 0)
        assert use_offline_data == 0
        super().__init__(**kargs)
        self.add_LunaSolarDirectly = kargs.get('add_LunaSolarDirectly', False)
        self.LaLotude, self.LaLotudeVector = self.get_mesh_lon_lat(32, 64)
        self.add_ConstDirectly = kargs.get('add_ConstDirectly', False)
        if self.add_ConstDirectly == 2:
            # we will normlize const, only the last one need
            mean = self.constants[-1].mean()
            std = self.constants[-1].std()
            self.constants[-1] = (self.constants[-1] - mean)/(std+1e-10)

    def config_pool_initial(self):
        self._component_list69 = self._component_list68 + [4]
        _list = self._component_list
        vector_scalar_mean = self.mean_std.copy()
        vector_scalar_mean[:,self.volicity_idx] = 0
        _list2 = [_list[iii] for iii in (list(range(0,14*4-1))+list(range(14*4,14*5-1)))]
        config_pool={
            '2D68S': (self._component_list68 ,'space_time_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
            '2D69S': (self._component_list69 ,'space_time_norm'  , self.mean_std[:,self._component_list69].reshape(2,69,1,1), identity, identity ),
        }
        return config_pool

    def get_mesh_lon_lat(self, tH=32, tW=64):
        resolution = tH
        assert tW == 2*tH
        theta_offset= (180/resolution/2)
        latitude   = (np.linspace(0,180,resolution+1) + theta_offset)[:resolution]
        longitude  = np.linspace(0,360,2*resolution+1)[:(2*resolution)]
        x, y       = np.meshgrid(latitude, longitude)
        LaLotude = np.stack([y, x])/180*np.pi
        LaLotudeVector = np.stack([np.cos(LaLotude[1])*np.cos(LaLotude[0]), np.cos(
            LaLotude[1])*np.sin(LaLotude[0]), np.sin(LaLotude[1])], 2)
        return LaLotude, LaLotudeVector
    
    def get_space_time_mean_std(self,idx):
        if self.space_time_mean is None:
            self.space_time_mean = np.zeros((8784,110,32,64))
            self.space_time_std  = np.zeros((8784,110,32,64))
            self.loaded_flag = {}
        now_time_stamp  = self.datatimelist_pool[self.split][idx]
        start_time_stamp= np.datetime64("2016-01-01")
        the_meanstd_index = int((now_time_stamp - start_time_stamp)/ np.timedelta64(1,'h'))
        the_meanstd_index =the_meanstd_index%8784
        if the_meanstd_index not in self.loaded_flag:
            self.loaded_flag[the_meanstd_index]=1
            self.space_time_mean[the_meanstd_index], self.space_time_std[the_meanstd_index] = np.load(
                f"datasets/weatherbench32x64/sp_meanstd/{the_meanstd_index:4d}.npy") # this will gradually take 28G memory
        return self.space_time_mean[the_meanstd_index][self.channel_choice], self.space_time_std[the_meanstd_index][self.channel_choice]

    def generate_runtime_data(self,idx,reversed_part=False):
        assert not reversed_part
        assert self.normalize_type == 'space_time_norm'
        odata = self.load_otensor(idx)[self.channel_choice]
        mean,std = self.get_space_time_mean_std(idx)
        data    = (odata - mean)/(std)
        if self.add_LunaSolarDirectly:
            timenow = self.datatimelist_pool[self.split][idx]
            moon_lon, moon_lat = get_sub_luna_point(timenow.item())
            sun_lon, sun_lat = get_sub_sun_point(timenow.item())
            sun_vector = np.stack([np.cos(sun_lat/180*np.pi)*np.cos(sun_lon/180*np.pi),
                                   np.cos(sun_lat/180*np.pi) *
                                   np.sin(sun_lon/180*np.pi),
                                   np.sin(sun_lat/180*np.pi)])
            moon_vector = np.stack([np.cos(moon_lat/180*np.pi)*np.cos(moon_lon/180*np.pi),
                                    np.cos(moon_lat/180*np.pi) *
                                    np.sin(moon_lon/180*np.pi),
                                    np.sin(moon_lat/180*np.pi)])
            sun_mask = (self.LaLotudeVector@sun_vector).reshape(1, 32, 64)
            moon_mask = (self.LaLotudeVector@moon_vector).reshape(1, 32, 64)
            data = np.concatenate([data, sun_mask, moon_mask])
        if self.add_ConstDirectly:
            data = np.concatenate([data, self.constants])
        
        return data
    
    def recovery(self,x,indexes):
        fake_mean, fake_std = self.mean_std
        
        real_mean = torch.from_numpy(np.stack(self.get_space_time_mean_std(idx)[0] for idx in indexes))  # (B, 68, 32, 64)
        real_std  = torch.from_numpy(np.stack(self.get_space_time_mean_std(idx)[1] for idx in indexes))  # (B, 68, 32, 64)

        x = x * real_std.to(x.device) + real_mean.to(x.device) # (B,68,32,64) * (B,68,32,64) + (B,68,32,64)
        x = x/torch.from_numpy(self.std[None]).to(x.device)
        return x

class WeathBench32x64Dailynorm(WeathBench32x64SPnorm):
    '''
    notice when generate daily norm, we firstly divide a unit to avoid overflow.
    '''
    def get_space_time_mean_std(self,idx):
        if self.space_time_mean is None:
            self.space_time_mean = np.zeros((8784,110,32,64))
            self.space_time_std  = np.zeros((8784,110,32,64))
            self.loaded_flag = {}
        now_time_stamp  = self.datatimelist_pool[self.split][idx]
        start_time_stamp= np.datetime64("2016-01-01")
        the_meanstd_index = int((now_time_stamp - start_time_stamp)/ np.timedelta64(1,'h'))
        the_meanstd_index = the_meanstd_index%8784
        if the_meanstd_index not in self.loaded_flag:
            self.loaded_flag[the_meanstd_index]=1
            mean, std = np.load(
                f"datasets/weatherbench32x64/daily_meanstd/{the_meanstd_index:4d}.npy").reshape(2,110,32,64) # this will gradually take 28G memory
            self.space_time_mean[the_meanstd_index] = mean
            std[std<0.01]=1
            self.space_time_std[the_meanstd_index]=std #we find some variable std is zero, then skip. The larget value in the zero slot is 2.7 safe for using
        
        return self.space_time_mean[the_meanstd_index][self.channel_choice], self.space_time_std[the_meanstd_index][self.channel_choice]
    
    def generate_runtime_data(self,idx,reversed_part=False):
        assert not reversed_part
        assert self.normalize_type == 'space_time_norm'
        odata    = self.load_otensor(idx)[self.channel_choice]
        unit    = self.std
        mean,std  = self.get_space_time_mean_std(idx)
        data    = (odata/unit - mean)/(std)
        if self.add_LunaSolarDirectly:
            timenow = self.datatimelist_pool[self.split][idx]
            moon_lon, moon_lat = get_sub_luna_point(timenow.item())
            sun_lon, sun_lat = get_sub_sun_point(timenow.item())
            sun_vector = np.stack([np.cos(sun_lat/180*np.pi)*np.cos(sun_lon/180*np.pi),
                                   np.cos(sun_lat/180*np.pi) *
                                   np.sin(sun_lon/180*np.pi),
                                   np.sin(sun_lat/180*np.pi)])
            moon_vector = np.stack([np.cos(moon_lat/180*np.pi)*np.cos(moon_lon/180*np.pi),
                                    np.cos(moon_lat/180*np.pi) *
                                    np.sin(moon_lon/180*np.pi),
                                    np.sin(moon_lat/180*np.pi)])
            sun_mask = (self.LaLotudeVector@sun_vector).reshape(1, 32, 64)
            moon_mask = (self.LaLotudeVector@moon_vector).reshape(1, 32, 64)
            data = np.concatenate([data, sun_mask, moon_mask])
        if self.add_ConstDirectly:
            data = np.concatenate([data, self.constants])
        
        return data

    def get_item(self,idx,reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        assert not self.use_offline_data
        data = self.generate_runtime_data(idx,reversed_part=reversed_part)
        
        if self.use_time_stamp:
            mean,std  = self.get_space_time_mean_std(idx)
            return data, mean*std
        else:
            return data


    def recovery(self,x,indexes):
        fake_mean, fake_std = self.mean_std
        
        real_mean = torch.from_numpy(np.stack(self.get_space_time_mean_std(idx)[0] for idx in indexes))  # (B, 68, 32, 64)
        real_std  = torch.from_numpy(np.stack(self.get_space_time_mean_std(idx)[1] for idx in indexes))  # (B, 68, 32, 64)

        x = x * real_std.to(x.device) + real_mean.to(x.device) # (B,68,32,64) * (B,68,32,64) + (B,68,32,64)
        #x = x/torch.from_numpy(self.std[None]).to(x.device)#notice when generate daily norm, then we wont divide this more.
        return x


class WeathBench32x64MultibranchRandom(WeathBench32x64):
    def __init__(self, **kargs):
        use_offline_data = kargs.get('use_offline_data', 0)
        time_intervel   = kargs.get('time_intervel', 1)
        assert time_intervel == 1
        assert use_offline_data == 0
        super().__init__(**kargs)
        self.multibranch_select= [int(t) for t in kargs['multibranch_select']]
    

    def __getitem__(self, idx):
        reversed_part = self.do_time_reverse(idx)
        time_intervel = self.time_intervel
        if self.split=='train':
            time_intervel = np.random.choice(self.multibranch_select)
            print(f"we use random time_intervel:{time_intervel}.However, this line should not appear here. You need call a random fetcher outside" )
        time_step_list = [idx+i*time_intervel for i in range(self.time_step)]
        if reversed_part:time_step_list = time_step_list[::-1]
        batch = [self.get_item(i, reversed_part) for i in time_step_list]
        self.error_path = []
        return batch if not self.with_idx else (idx, batch)

    
class WeathBench32x64CK(WeathBench):
    default_root = 'datasets/weatherbench32x64'
    
    def config_pool_initial(self):
        CK_order = [1, 2, 0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
               29, 30, 31, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 
               68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
        config_pool={
            'SWINRNN69' : (CK_order    ,'gauss_norm'   , self.mean_std[:,CK_order].reshape(2,69,1,1)      , identity, identity ),
        }
        self.constant_index = [0,2]
        return config_pool
        
    def load_numpy_from_url(self,url):#the saved numpy is not buffer, so use normal reading
        if "s3://" in url:
            if self.client is None:self.client=Client(conf_path="~/petreloss.conf")
            with io.BytesIO(self.client.get(url)) as f:
                array = np.load(f)
        else:
            array = np.load(url)
        return array
    
    def get_item(self,idx,reversed_part=False):
        year, hour = self.single_data_path_list[idx]
        url  = f"{self.root}/{year}/{year}-{hour:04d}.npy"
        odata = np.load(url)
        data = odata[self.channel_choice]
        data = (data - self.mean)/self.std
        cons = self.constants[self.constant_index]
        return np.concatenate([cons,data])
        
class WeathBench32x64d6(WeathBench71):
    def __len__(self):
        aaa =  len(self.single_data_path_list) - self.time_step*self.time_intervel + 1
        return aaa//self.time_intervel

    def __getitem__(self,idx):  
        time_step_list= [(idx+i)*self.time_intervel for i in range(self.time_step)]
        batch = [self.get_item(i,False) for i in time_step_list]
        self.error_path = []
        return batch if not self.with_idx else (idx,batch)
      
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
        print("you are trying to use 71 dataset with 6 time step, I recommend to use 7066 dataset for the whole project benchmark")
        raise
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
            print("notice the property 55 and 69 loss precision due to the .half operation. so please use use_offline_data=2 for inference.")
            self.use_offline_data = (kargs.get('split')=='train')
        if use_offline_data ==2:
            print("use offline data mode <2>: train/valid/test use offline data")
            self.use_offline_data = 2

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
        _list2 = [_list[iii] for iii in (list(range(0,14*4-1))+list(range(14*4,14*5-1)))]
        config_pool={
            '2D55N': (self._component_list55 ,'gauss_norm'  , self.mean_std[:,self._component_list55].reshape(2,55,1,1), identity, identity ),
            '2D68K': (self._component_list68 ,'gauss_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
            '2D68N': (self._component_list68 ,'gauss_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
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

class WeathBench64x128(WeathBench71):
    default_root='datasets/weatherbench64x128'
    use_offline_data = False
    time_unit=1
    years_split={'train':range(1979, 2016),
            'subtrain':range(2014, 2016),
            #'subtrain':range(2017, 2018),
            'valid':range(2016, 2018),
            'test':range(2018,2019),
            'ftest':range(1979,1980),
            'all': range(1979, 2022),
            'debug':range(1979,1980)}
    datatimelist_pool={'train':np.arange(np.datetime64("1979-01-01")+np.timedelta64(7, "h"), np.datetime64("2016-01-01"), np.timedelta64(1, "h")),
                       'valid':np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"), np.timedelta64(1, "h")),
                       'subtrain': np.arange(np.datetime64("2017-01-01"), np.datetime64("2018-01-01"), np.timedelta64(1, "h")),
                       'test':np.arange(np.datetime64("2018-01-01"), np.datetime64("2019-01-01"), np.timedelta64(1, "h")),
                       'ftest':np.arange(np.datetime64("1979-01-02"), np.datetime64("1980-01-01"), np.timedelta64(1, "h"))}
    def __init__(self,**kargs):
        use_offline_data = kargs.get('use_offline_data',0) 
        assert use_offline_data == 0
        super().__init__(**kargs)
        self.add_LunaSolarDirectly =  kargs.get('add_LunaSolarDirectly',False) 
        self.LaLotude,self.LaLotudeVector = self.get_mesh_lon_lat(64,128)
        self.add_ConstDirectly =  kargs.get('add_ConstDirectly',False) 
        if self.add_ConstDirectly == 2:
            # we will normlize const, only the last one need 
            mean = self.constants[-1].mean()
            std  = self.constants[-1].std()
            self.constants[-1] = (self.constants[-1] - mean)/(std+1e-10)
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        # not allow, it will take too much memory
        return None,None

    def config_pool_initial(self):
        _list = self._component_list
        vector_scalar_mean = self.mean_std.copy()
        vector_scalar_mean[:,self.volicity_idx] = 0
        _list2 = [_list[iii] for iii in (list(range(0,14*4-1))+list(range(14*4,14*5-1)))]
        config_pool={
            '2D55N': (self._component_list55 ,'gauss_norm'  , self.mean_std[:,self._component_list55].reshape(2,55,1,1), identity, identity ),
            '2D68K': (self._component_list68 ,'gauss_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
            '2D68N': (self._component_list68 ,'gauss_norm'  , self.mean_std[:,self._component_list68].reshape(2,68,1,1), identity, identity ),
            '2D70V': (_list ,'gauss_norm'   , vector_scalar_mean[:,_list].reshape(2,70,1,1), identity, identity ),
            #'2D70N': (_list ,'gauss_norm'   , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            #'2D70U': (_list ,'unit_norm'    , self.mean_std[:,_list].reshape(2,70,1,1), identity, identity ),
            #'3D70N': (_list ,'gauss_norm_3D', self.mean_std[:,_list].reshape(2,5,14,1,1), identity, identity ),
        }
        return config_pool

    def __len__(self):
        return (len(self.single_data_path_list) - self.time_step*self.time_intervel + 1)//2

    def generate_runtime_data(self,idx,reversed_part=False):
        odata=self.load_otensor(idx) #<-- we do normlization in load_otensor
        assert not reversed_part
        data = odata
        eg = torch if isinstance(data,torch.Tensor) else np
        return data
    
    def get_item(self,idx,reversed_part=False):
        data = self.generate_runtime_data(idx,reversed_part=reversed_part)
        if self.add_LunaSolarDirectly:
            timenow = self.datatimelist_pool[self.split][idx]
            moon_lon, moon_lat = get_sub_luna_point(timenow.item())
            sun_lon, sun_lat = get_sub_sun_point(timenow.item())
            sun_vector = np.stack([np.cos(sun_lat/180*np.pi)*np.cos(sun_lon/180*np.pi),
                        np.cos(sun_lat/180*np.pi)*np.sin(sun_lon/180*np.pi),
                        np.sin(sun_lat/180*np.pi)])
            moon_vector = np.stack([np.cos(moon_lat/180*np.pi)*np.cos(moon_lon/180*np.pi),
                        np.cos(moon_lat/180*np.pi)*np.sin(moon_lon/180*np.pi),
                        np.sin(moon_lat/180*np.pi)])
            sun_mask = (self.LaLotudeVector@sun_vector).reshape(1,64,128)
            moon_mask = (self.LaLotudeVector@moon_vector).reshape(1,64,128)
            data = np.concatenate([data,sun_mask,moon_mask])
        if self.add_ConstDirectly:
            data = np.concatenate([data,self.constants])
        
        return data

    def load_otensor(self, idx):
        if (self.record_load_tensor is None) or (not self.record_load_tensor[idx]):
            year, hour = self.single_data_path_list[idx]
            url = f"{self.root}/{year}/{year}-{hour:04d}.npy"
            # let's do feature pick here, to save memory.
            odata = self.load_numpy_from_url(url)[self.channel_choice]
            odata = (odata - self.mean)/self.std
            if self.record_load_tensor is not None:
                self.record_load_tensor[idx] = 1
                self.dataset_tensor[idx] = torch.Tensor(odata)
        if self.record_load_tensor is not None:
            return self.dataset_tensor[idx]
        else:
            return odata
        
class WeathBench64x128CK(WeathBench64x128):
    
    def config_pool_initial(self):
        CK_order = [1, 2, 0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
               29, 30, 31, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 
               68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
        config_pool={
            'SWINRNN69' : (CK_order    ,'gauss_norm'   , self.mean_std[:,CK_order].reshape(2,69,1,1)      , identity, identity ),
        }
        self.constant_index = [0,2]
        return config_pool
    
    def get_item(self,idx,reversed_part=False):
        odata = self.load_otensor(idx)
        data = odata #[self.channel_choice] we do channel pick in self.load_otensor

        cons = self.constants[self.constant_index]
        return np.concatenate([cons,data])


class WeathBench128x256(WeathBench64x128):
    default_root='datasets/weatherbench128x256'

class WeathBench64x128CK(WeathBench64x128CK):
    default_root='datasets/weatherbench128x256'


class WeathBenchUpSize_64_to_128:
    def __init__(self, dataset_tensor_1=None,record_load_tensor_1=None,dataset_tensor_2=None,record_load_tensor_2=None,**kargs):
        assert kargs['root']
        newkargs = copy.deepcopy(kargs)
        newkargs['split']='subtrain'
        newkargs['root'] = kargs['root'].replace('32x64','64x128').replace('128x256','64x128')
        
        self.dataset_64x128 = WeathBench64x128CK(dataset_tensor=dataset_tensor_1,record_load_tensor=record_load_tensor_1,**newkargs)
        newkargs['root'] = kargs['root'].replace('32x64','128x256').replace('64x128','128x256')
        self.dataset_128x256 = WeathBench64x128CK(dataset_tensor=dataset_tensor_2,record_load_tensor=record_load_tensor_2,**newkargs)
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
        years = WeathBench64x128.years_split['subtrain']
        batches = len(WeathBench64x128.init_file_list(years))
        if split == 'train':
            return torch.empty(batches, 69, 64, 128), torch.zeros(batches)
        elif split == 'valid':
            return torch.empty(batches, 69, 128, 256), torch.zeros(batches)
        else:
            raise
    
class WeathBench7066Self(WeathBench7066):
    # use for property relation check
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        picked_input_property  = kargs.get('picked_input_property') 
        picked_output_property = kargs.get('picked_output_property') 

        assert picked_input_property
        assert picked_output_property
        if isinstance(picked_input_property,int):picked_input_property = [picked_input_property]
        if isinstance(picked_output_property,int):picked_output_property = [picked_output_property]

        picked_input_property = set(picked_input_property)
        picked_output_property= set(picked_output_property)

        assert len(picked_input_property&picked_output_property) == 0
        self.picked_input_channel = []
        for p in picked_input_property:
            self.picked_input_channel += list(range(p*14,(p+1)*14))

        self.picked_output_channel = []
        for p in picked_output_property:
            self.picked_output_channel += list(range(p*14,(p+1)*14))

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self,idx):
        batch = self.get_item(idx)
        batch = [batch[self.picked_input_channel],batch[self.picked_output_channel]]
        return batch if not self.with_idx else (idx,batch)
  
class WeathBench7066deseasonal(WeathBench7066):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        assert self.use_time_stamp 
        self.deseasonal_mean, self.deseasonal_std = self.load_numpy_from_url(os.path.join(self.root+"_offline","mean_stds_deseasonal.npy"))
        self.deseasonal_mean = self.deseasonal_mean.reshape(70,1,1)
        self.deseasonal_std  = self.deseasonal_std.reshape(70,1,1)
        self.deseasonal_mean_tensor = torch.Tensor(self.deseasonal_mean).reshape(1,70,1,1)
        self.deseasonal_std_tensor  = torch.Tensor(self.deseasonal_std).reshape(1,70,1,1)
        self.seasonal_tensor        = torch.Tensor(np.load("datasets/weatherbench_6hour_offline/seasonal1461.npy"))
        time_stamps              = self.datatimelist_pool[self.split]
        sean_start_stamps = np.datetime64("1979-01-02")
        offset            = ((time_stamps - sean_start_stamps)% (1461*np.timedelta64(6, "h")))//np.timedelta64(6, "h")
        self.timestamp    = torch.LongTensor(offset)
    def addseasonal(self, tensor, time_stamps_offset):
        if self.seasonal_tensor.device != tensor.device:
            self.deseasonal_std_tensor  = self.deseasonal_std_tensor.to(tensor.device)
            self.deseasonal_mean_tensor = self.deseasonal_mean_tensor.to(tensor.device)
            self.seasonal_tensor = self.seasonal_tensor.to(tensor.device)
        tensor = tensor*self.deseasonal_std_tensor + self.deseasonal_mean_tensor
        tensor = tensor + self.seasonal_tensor[time_stamps_offset.long()] #(B, 70, 32, 64)
        return tensor

    def recovery(self,tensor, time_stamps_offset):
        return self.addseasonal(tensor, time_stamps_offset)
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:root = WeathBench7066.default_root
        if use_offline_data:
            dataset_flag = kargs.get('dataset_flag')
            data_name = f"{split}_{dataset_flag}_deseasonal.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root+'_offline',data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor   = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor

class WeathBench68pixelnorm(WeathBench7066):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        assert self.dataset_flag == '2D68K'
        self.pixelnorm_mean, self.pixelnorm_std = self.load_numpy_from_url(os.path.join(self.root+"_offline","means_stds_pixelnorm.npy"))
        self.pixelnorm_mean_tensor  = torch.Tensor(self.pixelnorm_mean).reshape(1,68,32,64)
        self.pixelnorm_std_tensor  = torch.Tensor(self.pixelnorm_std).reshape(1,68,32,64)
        
    def recovery(self, tensor):
        if self.pixelnorm_mean_tensor.device != tensor.device:
            self.pixelnorm_std_tensor  = self.pixelnorm_std_tensor.to(tensor.device)
            self.pixelnorm_mean_tensor = self.pixelnorm_mean_tensor.to(tensor.device)
        tensor = tensor*self.pixelnorm_std_tensor + self.pixelnorm_mean_tensor
        return tensor
    
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:root = WeathBench7066.default_root
        if use_offline_data:
            dataset_flag = kargs.get('dataset_flag')
            data_name = f"{split}_{dataset_flag}_pixelnorm.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root+'_offline',data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor   = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor

class WeathBench55withoutH(WeathBench7066):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        assert self.dataset_flag == '2D55N'
        self.dataset_tensor = self.dataset_tensor[:,:55]
        #print(self.dataset_tensor.shape)
    
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:root = WeathBench7066.default_root
        if use_offline_data:
            dataset_flag = "2D70N"
            data_name = f"{split}_{dataset_flag}.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root,data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor   = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor

class WeathBench69SolarLunaMask(WeathBench7066):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        assert self.dataset_flag   == '2D68N'
        pick_list = list(range(55)) + list(range(56,69))
        self.dataset_tensor      = self.dataset_tensor[:,pick_list]
        self.LaLotude,self.LaLotudeVector = self.get_mesh_lon_lat()
    
    @staticmethod
    def create_offline_dataset_templete(split='test', root=None, use_offline_data=False, **kargs):
        if root is None:root = WeathBench7066.default_root
        if use_offline_data:
            dataset_flag = "2D70N"
            data_name = f"{split}_{dataset_flag}.npy"
        else:
            raise NotImplementedError
        numpy_path = os.path.join(root,data_name)
        print(f"load data from {numpy_path}")
        dataset_tensor   = torch.Tensor(np.load(numpy_path))
        record_load_tensor = torch.ones(len(dataset_tensor))
        return dataset_tensor,record_load_tensor

    def get_item(self,idx,reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        assert self.use_offline_data==2
        data =  self.dataset_tensor[idx]
        timenow = self.datatimelist_pool[self.split][idx]
        moon_lon, moon_lat = get_sub_luna_point(timenow.item())
        sun_lon, sun_lat = get_sub_sun_point(timenow.item())
        sun_vector = np.stack([np.cos(sun_lat/180*np.pi)*np.cos(sun_lon/180*np.pi),
                    np.cos(sun_lat/180*np.pi)*np.sin(sun_lon/180*np.pi),
                    np.sin(sun_lat/180*np.pi)])
        moon_vector = np.stack([np.cos(moon_lat/180*np.pi)*np.cos(moon_lon/180*np.pi),
                    np.cos(moon_lat/180*np.pi)*np.sin(moon_lon/180*np.pi),
                    np.sin(moon_lat/180*np.pi)])
        sun_mask = torch.Tensor(self.LaLotudeVector@sun_vector).reshape(1,32,64)
        moon_mask = torch.Tensor(self.LaLotudeVector@moon_vector).reshape(1,32,64)
        data = [data,sun_mask,moon_mask]

        return data

class WeathBench7066DeltaDataset(WeathBench7066):
    def __init__(self,**kargs):
        super().__init__(**kargs)
        assert self.use_offline_data == 2
        self.delta_mean, self.delta_std = self.load_numpy_from_url(os.path.join(self.root,"delta_mean_std.npy"))
        self.delta_mean = self.delta_mean.reshape(70,1,1)
        self.delta_std  = self.delta_std.reshape(70,1,1)
        self.delta_mean_tensor = torch.Tensor(self.delta_mean).reshape(1,70,1,1)
        self.delta_std_tensor = torch.Tensor(self.delta_std).reshape(1,70,1,1)
    def __len__(self):
        return len(self.dataset_tensor) - self.time_step*self.time_intervel + 1 -1

    def combine_base_delta(self,base, delta):
        if self.delta_std_tensor.device != delta.device:
            self.delta_std_tensor = self.delta_std_tensor.to(delta.device)
        return  base + delta*self.delta_std_tensor + self.delta_mean_tensor
    
    def recovery(self,base, delta):
        return self.combine_base_delta(base, delta)

    def get_item(self,idx,reversed_part=False):
        '''
        Notice for 3D case, we return (5,14,32,64) data tensor
        '''
        assert self.use_offline_data
        data  =  self.dataset_tensor[idx]
        delta  = (self.dataset_tensor[idx + 1] - data)
        delta = (delta - self.delta_mean)/self.delta_std
        return data,delta

class WeathBench7066PatchDataset(WeathBench7066):
    def __init__(self,**kargs):
        self.use_offline_data = kargs.get('use_offline_data',0) and kargs.get('split')=='train'
        super().__init__(**kargs)
        self.cross_sample     = kargs.get('cross_sample', True) and ((self.split == 'train') or (kargs.get('debug', 0)))
        
        patch_range = kargs.get('patch_range',5)
        
        #self.img_shape        = kargs.get('img_size',WeathBench7066PatchDataset.img_shape)
        #if isinstance(self.img_shape,str):self.img_shape=tuple([int(p) for p in self.img_shape.split(',')])
        self.img_shape       = WeathBench7066PatchDataset.img_shape
        #print(self.img_shape)
        

        if '3D' in self.normalize_type:
            #self.img_shape        = kargs.get('img_size',WeathBench7066PatchDataset.img_shape)
            #if isinstance(self.img_shape,str):self.img_shape=tuple([int(p) for p in self.img_shape.split(',')])
            self.img_shape = (14,32,64)
            patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range,patch_range)
            self.center_index,self.around_index = get_center_around_indexes_3D(patch_range,self.img_shape)
        else:
            patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range)
            self.center_index,self.around_index = get_center_around_indexes(patch_range,self.img_shape)
        print(f"notice we will use around_index{self.around_index.shape} to patch data")
        self.channel_last                   = False
        #self.random = kargs.get('random_dataset', False)
        self.use_position_idx = kargs.get('use_position_idx', False)
        self.patch_range      = patch_range
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
                patch_idx_z, patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_z, patch_idx_h, patch_idx_w]
            else:
                patch_idx_h, patch_idx_w = location
                data = data[..., patch_idx_h, patch_idx_w]
        else:
            location = -1
        
        
        out = [data]
        if self.use_time_stamp:out.append(self.timestamp[idx])
        if self.use_position_idx:out.append(location)
        if len(out)==1:out=out[0]
        return out

    def __getitem__(self,idx):
        #if self.random:idx = np.random.randint(self.__len__())
        reversed_part = self.do_time_reverse(idx)
        time_step_list= [idx+i*self.time_intervel for i in range(self.time_step)]
        if reversed_part:time_step_list = time_step_list[::-1]
        patch_idx_h = patch_idx_w = patch_idx_z = None
        location = None
        if self.cross_sample:
            center_h = np.random.randint(self.img_shape[-2] - (self.patch_range[-2]//2)*2) 
            center_w = np.random.randint(self.img_shape[-1])
            if '3D' in self.normalize_type:
                center_z      = np.random.randint(self.img_shape[-3] - (self.patch_range[-3]//2)*2) 
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
    dataset = WeathBench64x128(split='test',root='weatherbench:s3://weatherbench/weatherbench64x128/npy',dataset_flag='2D68N')
    for i in tqdm(range(0, 1000)):
        a = dataset[i][0]

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
