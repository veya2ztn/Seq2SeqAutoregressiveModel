
from .convNd import convNd
from collections import OrderedDict
import torch.nn as nn

import numpy as np

import time
class Timer:
    def __init__(self,active=True):
        self.recorder=OrderedDict()
        self.real_name=OrderedDict()
        self.father=OrderedDict()
        self.child =OrderedDict()
        self.last_time={}
        self.active = active
    def restart(self,level=0):

        self.last_time[level] = time.time()
    def record(self, name,father=None,level=0):
        if not self.active:return
        cost= time.time()- self.last_time[level]

        if name not in self.recorder:self.recorder[name]=[]
        self.recorder[name].append(cost)
        if father is not None:
            if father not in self.child:self.child[father]=[]
            if name not in self.child[father]:
                self.child[father].append(name)
            self.father[name]=father
        self.real_name[name]=name
        self.last_time[level] = time.time()

    def show_stat_per_key(self,key, level=0):
        print("--"*level+f"[{self.real_name[key]}]:cost {np.mean(self.recorder[key][1:]):.1e} ± {np.std(self.recorder[key][1:]):.1e}")
        if key not in self.child:return
        level=level+1
        for child in self.child[key]:
            self.show_stat_per_key(child,level)

    def show_stat(self):
        if not self.active:return
        for key in self.recorder.keys():
            if key in self.father:continue
            print(">"+f"[{key}]:cost {np.mean(self.recorder[key][1:]):.1e} ± {np.std(self.recorder[key][1:]):.1e}")
            if key not in self.child:continue
            for child in self.child[key]:
                self.show_stat_per_key(child,1)
timer = Timer(False)

def transposeconv_engines(dim, conv_simple=True):
    if dim in [1,2,3] and conv_simple:return [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][dim-1]
    return lambda *args,**kargs:convNd(*args,**kargs,num_dims=dim,is_transposed=True,use_bias=False)

def conv_engines(dim, conv_simple=True):
    if dim in [1,2,3] and conv_simple:return [nn.Conv1d,nn.Conv2d,nn.Conv3d][dim-1]
    return lambda *args,**kargs:convNd(*args,**kargs,num_dims=dim,is_transposed=False,use_bias=False)
