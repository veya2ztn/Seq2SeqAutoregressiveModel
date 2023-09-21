from dataclasses import dataclass, field
from typing import Optional, Union, List,Tuple
from .base import Config


@dataclass
class EngineConfig(Config):
    local_rank:int = field(default=0)       
    rank:int = field(default=0)
    world_size:int = field(default=1)  
    find_unused_parameters: bool = field(default=False)
    
    
@dataclass
class NaiveDistributed(EngineConfig):
    use_amp:bool = field(default=False)
    distributed:bool = field(default=False)
    torch_compile:bool = field(default=False)
    dist_backend: str = field(default="nccl")
    ngpus_per_node: int = field(default=0)
    
    @property
    def name(self):
        return 'naive_distributed'

@dataclass
class AccelerateEngine(EngineConfig):
    data_parallel_dispatch: bool = field(default=False)
    num_max_checkpoints: int = field(default=1)
    @property
    def name(self):
        return 'accelerate'
