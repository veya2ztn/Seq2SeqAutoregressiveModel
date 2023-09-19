from dataclasses import dataclass, field
from typing import Optional, Union, List,Tuple
from configs.base import Config

@dataclass    
class PatchEmbeddingConfig(Config):
    patch_size:Optional[Union[List[int], int]] = field(default = 2)
    

@dataclass
class ModelConfig(Config):
    img_size: List[int] = field(default_factory=lambda: [16, 32])
    history_length: int = field(default=1)
    in_chans: int = field(default=70)
    out_chans: int = field(default=70)
    embed_dim: int = field(default=768)
    depth: int = field(default=12)
    num_heads: int = field(default=16)
    
    def get_name(self):
        shape_string = "_".join([str(self.history_length)] + [str(t) for t in self.img_size])
        return f'{self.model_type}.{shape_string}.{self.in_chans}.{self.out_chans}.{self.embed_dim}.{self.depth}.{self.num_heads}'

@dataclass
class AFNONetConfig(ModelConfig):
    model_type: str = 'afnonet'
    drop_path_rate: float = field(default = 0)
    mlp_ratio:float = field(default = 0)
    drop_rate:float = field(default = 0)
    double_skip:bool = field(default = False)
    fno_bias:bool = field(default = False)
    fno_softshrink:bool = field(default = False)
    uniform_drop:bool = field(default = False)


@dataclass
class GraphCastConfig(ModelConfig):
    model_type: str = 'graphcast'
    graphflag: str = field(default = 'mesh5')
    agg_way  : str = field(default = 'mean')
    nonlinear: str = field(default = 'swish')
