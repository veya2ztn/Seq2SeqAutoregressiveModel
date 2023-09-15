#from tkinter.messagebox import NO


import os

import pandas as pd
from tqdm import tqdm
import traceback

import numpy as np
import torch
import os
import io
import socket
from torchvision import datasets, transforms
from .utils import load_numpy_from_url
hostname = socket.gethostname()
#if hostname not in ['SH-IDC1-10-140-0-184','SH-IDC1-10-140-0-185'] and '54' not in hostname and '52' not in hostname:
try:
    from petrel_client.client import Client
    import petrel_client
except:
    Client = None
import typing
from dataclasses import dataclass, field


@dataclass
class DatasetArguments:
    root:str 
    time_unit:int
    resolution_w :int   
    resolution_h :int   
    channel_name_list:str
    timestamps_list:typing.Optional[list] = field(default=None)
    
    time_step    :int = field(default=2)    
    time_intervel:int = field(default=1)    
    patch_range: typing.Union[list, int, None] = field(default=None)
    
    normlized_flag                 : str  = field( default = 'N')
    time_reverse_flag              : str  = field( default ='only_forward')
    use_time_feature               : bool = field( default = False)
    add_LunaSolarDirectly          : bool = field( default = False)
    offline_data_is_already_normed : bool = field( default = False)
    cross_sample                   : bool = field( default = False)
    
    constant_channel_pick              : typing.Optional[list] = field(default=None)
    make_data_physical_reasonable_mode : typing.Optional[str] = field(default=None)

    def get(self, attribute, default=None):
        return getattr(self, attribute, default)
    
class BaseDataset:
    client = None
    time_intervel = 1
    error_path       = []
    retry_limit = 1
    def do_normlize_data(self, batch):
        raise NotImplementedError("Not use anymore")

    def inv_normlize_data(self, batch):
        raise NotImplementedError("Not use anymore")

        
    def load_numpy_from_url(self, url):
        if "s3://" in url:
            if self.client is None:
                assert Client is not None, "we must have valid Client"
                self.client = Client(conf_path="~/petreloss.conf")
        array = load_numpy_from_url(self.client, url)

        return array
    
    def do_time_reverse_augmentation_Q(self):
        ###### may get error when use distributed dataset as the function is not allowed to pickle.
        if self.time_reverse_flag == 'only_forward' :
           return False
           #print("we only using forward sequence, i.e. from t1, t2, ..., to tn")
        elif self.time_reverse_flag == 'only_backward':
           return self.volicity_idx
           #print("we only using backward sequence, i.e. from tn, tn-1, ..., to t1")
        elif self.time_reverse_flag == 'random_forward_backward':
            if np.random.random() > 0:
               return False
            else:
                return self.volicity_idx
           #print("we randomly(50%/50%) use forward/backward sequence")
        else:
           raise NotImplementedError
        # assert time_reverse_flag == 'only_forward'
        # self.do_time_reverse = getFalse

    def do_time_reverse(self, idx):
        return False

    def get_item(self,idx,*args):
        raise NotImplementedError("You must implement it")
    
    def __getitem__(self, idx):

        #try:
        reversed_part  = self.do_time_reverse_augmentation_Q()
        time_step_list = [idx+i*self.time_intervel for i in range(self.time_step)]
        # time reverse require the 
        # 1. reverse of the sequence
        # 2. reverse of the volecity
        if reversed_part:time_step_list = time_step_list[::-1]
        batch = [self.get_item(i, reversed_part) for i in time_step_list]
        dict_list =[]
        for i,data in enumerate(batch):
            _dict = {'field':data} if isinstance(data, (torch.Tensor, np.ndarray)) else data
            assert isinstance(_dict, dict)
            if self.with_idx:_dict['idx'] = idx
            dict_list.append(_dict)
        return dict_list
        # except:
        #     self.error_path.append(idx)
        #     if len(self.error_path) < self.retry_limit:
        #         next_idx = np.random.randint(0, len(self))
        #         return self.__getitem__(next_idx)
        #     else:
        #         print(self.error_path)
        #         traceback.print_exc()
        #         raise NotImplementedError("too many error happened, check the errer path")
