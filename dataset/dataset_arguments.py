from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from simple_parsing import  field
from typing import Optional, Union, List,Tuple
from configs.base import Config
import os

@dataclass
class DatasetConfig(Config):
    num_workers: int = field(default=0)
    root: str = field(default='datasets/WeatherBench/weatherbench32x64_1hour/')
    time_unit: int = field(default=1)
    dataset_patch_range: Optional[List[int]] = field(default=None)
    timestamps_list: int = field(default=None)
    time_step: int = field(default=2)
    time_intervel: int = field(default=1)
    share_memory: bool = field(default=False, help='share_memory_flag')
    
    def get_name(self):
        raise NotImplementedError
    
    @property
    def dataset_type(self):
        raise NotImplementedError

@dataclass
class WeatherBenchConfig(DatasetConfig):
    
    constant_channel_pick: Optional[List[int]] = field(default=None)
    channel_name_list: str = field(default="configs/datasets/WeatherBench/2D70.channel_list.json")
    normlized_flag: str = field(default='N')
    time_reverse_flag: str = field(default='only_forward')
    use_time_feature: bool = field(default=False)
    add_LunaSolarDirectly: bool = field(default=False)
    offline_data_is_already_normed: bool = field(default=False)
    make_data_physical_reasonable_mode: str = field(default=None)
    picked_inputoutput_property: str = field(default=None)
    random_time_step:bool = field(default=None)
    
    @property
    def dataset_type(self):
        return 'WeatherBench'
    def get_name(self):
        channel_name = os.path.split(self.channel_name_list)[-1].split('.')[0]
        return f'{self.name}.{channel_name}{self.normlized_flag}.per{self.time_intervel}.unit{self.time_unit}'

@dataclass
class WeatherBenchPatchConfig(WeatherBenchConfig):
    
    data_patch_range: Optional[List[int]] = field(default_factory=lambda: [5, 5])
    patch_chunk: int = field(default=128)
    def get_name(self):
        raise NotImplementedError

    @property
    def dataset_type(self):
        return 'WeatherBenchPatch'
