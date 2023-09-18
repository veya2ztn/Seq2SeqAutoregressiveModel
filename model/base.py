from typing import List, Union, Tuple, Optional
from typing import Optional, Tuple, Union, List, Dict, Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
  


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        #### common model config
        self.config = config
        self.img_size = config.img_size
        self.history_length = config.history_length
        self.patch_size = config.patch_size
        self.in_chans = config.in_chans
        self.out_chans = config.out_chans
        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.debug_mode = config.debug_mode

        self.patch_size = [self.patch_size]*len(self.img_size) if isinstance(self.patch_size, int) else self.patch_size

        if self.history_length > 1:
            self.img_size = (self.history_length,*self.img_size)
            self.patch_size = (1,*self.patch_size)

        # print(f"""
        # ============model:DownAndUp================
        #         img_size:{img_size}
        #         patch_size:{patch_size}
        #         in_chans:{in_chans}
        #         out_chans:{out_chans}
        # ========================================
        # """)
    def set_epoch(self,**kargs):
        pass
    
    def set_step(self,**kargs):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.constant_(m.weight,0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            #torch.nn.init.constant_(m.weight,0.5)
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 0.5)
            if m.bias is not None:
                m.bias.data.fill_(0)
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                    #torch.nn.init.constant_(param.data,0.5)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                    #torch.nn.init.constant_(param.data,0.5)
                elif 'bias' in name:
                    param.data.fill_(0)

    def get_w_resolution_pad(self, x):
        shape = x.shape
        w_now = shape[-2]
        w_should = self.img_size[-2]
        if w_now == w_should:
            return x, None
        if w_now > w_should:
            raise NotImplementedError
        if w_now < w_should and not ((w_should - w_now) % 2):
            # we only allow symmetry pad
            pad = (w_should - w_now)//2
            return F.pad(x.flatten(0, 1), (0, 0, pad, pad), mode='replicate').reshape(*shape[:-2], -1, shape[-1]), pad
        return x, pad

    def collect_correct_input(self, _input: List[Dict]):
        assert len(_input) == self.history_length
        if self.history_length == 1:
            x = _input[0]['field']
        else:
            # [(B,P,H,W),(B,P,H,W)] -> (B,P,L,H,W)
            x = torch.stack([_inp['field'] for _inp in _input], 2)
        # <--- in case the height is smaller than the img_size
        x, pad = self.get_w_resolution_pad(x)
        self.pad = pad
        return x

    def collect_correct_output(self, x: torch.Tensor):
        pad = self.pad
        if pad is not None:
            x = x[..., pad:-pad, :]
        return x
    
class DownAndUpModel(BaseModel):
    def forward_features(self, x):
        x = self.compress(x)
        x = self.timepostion_information(x)
        x = self.kernel(x)
        return x

    def forward(self, x: List[Dict], return_feature=False) -> Dict:
        ### we assume always feed the tensor (B, p*z, h, w)
        output = {}
        x = self.collect_correct_input(x)
        x = self.forward_features(x)
        if return_feature:
            output['feature'] = x
        x = self.decompress(x)
        x = self.collect_correct_output(x)
        output['field'] = x
        return output

class Sphere_Model(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone = backbone
        self.block_target_timestamp = args.block_target_timestamp
        self.timespace_shape = (args.history_length, *args.img_size)
        self.time_length = args.history_length
        self.space_shape = args.img_size
        self.pred_len = args.pred_len

    def get_direction_from_time_stamp(self, x_timestamp):
        x_timestamp = x_timestamp*np.pi  # the input is in [-1,1]
        if x_timestamp.shape[-1] == 4:
            # if the time feature is 4 then only the first and last is needed
            x_timestamp = x_timestamp[..., [0, -1]]
        year_pos_x = x_timestamp[..., 0]
        day_pos_x = x_timestamp[..., 1]
        x_direction = torch.stack([torch.cos(year_pos_x),
                                   torch.sin(year_pos_x)*torch.cos(day_pos_x),
                                   torch.cos(year_pos_x)*torch.sin(day_pos_x)], 1)  # (B, 3 ,T)
        return x_direction

class Time_Sphere_Model(Sphere_Model):
    '''
    x_data is the Field data without time information as (B, T, h ,w,  P)
    x_timestamp is the stamp vector via time_features
     it is a (B, T, 2) data, represent which day in year, which hour in the day,  
    the feature will be projected into a sphere coordinate by use the 
    [Pcos(year_pos),Psin(year_pos)cos(day_pos),Psin(year_pos)cos(day_pos)]
    so the final tensor is (B, T, h ,w,  P, 3) --> (B, T, h ,w,  3P)
    ------------------------------------------------------------------------
    after passing this tensor into the backbone 
    (B, T, h ,w, 3P) --> backbone --> (B, T, h ,w, 3P) --> (B, T, h ,w, P,3)
    ------------------------------------------------------------------------
    we will do the norm operation at end, and 
    norm( [B, T, h ,w, P,3] ) -->  [B, T, h ,w, P] 
    ---------------------------------------------------------------------------
    if the y_timestamp is given, then extraloss to reforce the sphere constrain.
      cos([out1,out2,out3], [cos(year_pos),sin(year_pos)cos(day_pos),sin(year_pos)cos(day_pos)])must equal 1
    '''

    def forward(self, x, x_timestamp, y_timestamp):
        # assume input is (B, P ,T, h ,w)
        assert x.shape[2:] == self.timespace_shape
        assert x_timestamp.shape[1] == self.time_length
        B, P = x.shape[:2]

        x_direction = self.get_direction_from_time_stamp(x_timestamp)
        y_direction = self.get_direction_from_time_stamp(y_timestamp)
        x = torch.einsum('bpt...,bdt->bdpt...', x, x_direction).flatten(1, 2)
        x = self.backbone(x)
        x = x.reshape(B, 3, P, self.pred_len, *self.space_shape)

        extra_loss = 0
        if not self.block_target_timestamp:
            Pshape = [B, 3]+[1]*(len(x.shape)-2)
            y_direction = y_direction.reshape(*Pshape)
            extra_loss = 1 - \
                torch.nn.functional.cosine_similarity(
                    x, y_direction, dim=1).mean()
        x = x.norm(dim=1)
        return x, extra_loss

class Time_Projection_Model(Sphere_Model):
    extra_loss_coef = 1
    '''
    x_data is the Field data without time information as (B, T, h ,w,  P)
    x_timestamp is the stamp vector via time_features
     it is a (B, T, 2) data, represent which day in year, which hour in the day,  
    the feature will be projected into a sphere coordinate by use the 
    [Pcos(year_pos),Psin(year_pos)cos(day_pos),Psin(year_pos)cos(day_pos)]
    so the final tensor is (B, T, h ,w,  P, 3) --> (B, T, h ,w,  3P)
    ------------------------------------------------------------------------
    after passing this tensor into the backbone 
    (B, T, h ,w, 3P) --> backbone --> (B, T, h ,w, 3P) --> (B, T, h ,w, P,3)
    ------------------------------------------------------------------------
    we will project this vector into the time vector [cos(year_pos),sin(year_pos)cos(day_pos),sin(year_pos)cos(day_pos)]
    [B, T, h ,w, P, 3] dot n 
    and get the 
    [B, T, h ,w, P, 3]
    mean while the self projection is required also
    ---------------------------------------------------------------------------
    '''

    def set_epoch(self, epoch, epoch_total):
        self.extra_loss_coef = (
            np.exp(1) - np.exp(epoch/epoch_total))/(np.exp(1) - 1)

    def forward(self, x, x_timestamp, y_timestamp):
        # assume input is (B, P ,T, h ,w)
        assert self.pred_len == 1
        assert x.shape[2:] == self.timespace_shape
        assert x_timestamp.shape[1] == self.time_length
        B, P = x.shape[:2]

        x_direction = self.get_direction_from_time_stamp(x_timestamp)
        y_direction = self.get_direction_from_time_stamp(y_timestamp)
        y = torch.einsum('bpt...,bdt->bdpt...', x, x_direction).flatten(1, 2)
        y = self.backbone(y)
        y = y.reshape(B, 3, P, *self.space_shape)
        # should be (B, 3, P,self.pred_len,  *self.space_shape). omit self.pred_len since it is set 1
        self_projection = torch.einsum('bdp...,bdt->bpt...', y, x_direction)

        extra_loss = F.mse_loss(self_projection, x)

        target_projection = torch.einsum('bdp...,bdt->bpt...', y, y_direction)
        return target_projection, self.extra_loss_coef*extra_loss

class LoRALinear(nn.Module):
        def __init__(self, in_channel,out_channel):
            super().__init__()
            self.main = torch.nn.Linear(in_channel,out_channel,bias=False)
            self.lora = None
            # we will assign lora via outside function
        def forward(self,x):
            if self.lora is None:
                return self.main(x)
            else:
                assert not self.main.weight.requires_grad
                return self.main(x) + self.lora(x)

