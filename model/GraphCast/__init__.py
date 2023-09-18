from .GraphCast import GraphCastFast
from ..base import BaseModel

class GraphCastModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        #assert config.history_length == 1, "GraphCast only support history_length == 1"
        self.backbone = GraphCastFast(config)
    
    def collect_correct_input(self, _input: List[Dict]):
        assert len(_input) == self.history_length ## <--- if multiple timestamp is input, concat the channel
        if len(_input)>1:
            x = torch.cat([_inp['field'] for _inp in _input], 1) #(B, n*P, H, W)
        else:
            x = _input[0]['field']
        ## << Should fullfill the inchannel requirement!!!
        
        # <--- in case the height is smaller than the img_size
        x, pad = self.get_w_resolution_pad(x)
        self.pad = pad
        return x

    def forward(self, x: List[Dict], return_feature=False) -> Dict:
        ### we assume always feed the tensor (B, p*z, h, w)
        output = {}
        x = self.collect_correct_input(x)
        x = self.backbone(x, return_feature=return_feature)
        if return_feature:
            output['feature'] = x[1]
            x = x[0]
        output['field'] = x
        return output
    