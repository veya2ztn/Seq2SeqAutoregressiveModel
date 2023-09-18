from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple


class Config:
    def get_item(self,idx,*args):
        raise NotImplementedError("You must implement it")

@dataclass
class AFNONetConfig(Config):
    drop_path_rate: float = field(default = 0)
    mlp_ratio:float = field(default = 0)
    drop_rate:float = field(default = 0)
    double_skip:bool = field(default = False)
    fno_bias:bool = field(default = False)
    fno_softshrink:bool = field(default = False)
    uniform_drop:bool = field(default = False)


@dataclass    
class PatchEmbeddingConfig(Config):
    patch_size:Optional[Union[List[int], int]] = field(default = 2)


@dataclass
class GraphCastConfig(Config):
    graphflag: str = field(default = 'mesh5')
    agg_way  : str = field(default = 'mean')
    nonlinear: str = field(default = 'swish')
