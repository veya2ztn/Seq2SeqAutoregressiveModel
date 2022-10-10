import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch
from .afnonet import BaseModel

class Time_Sphere_Model(BaseModel):
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
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
        self.block_target_timestamp = args.block_target_timestamp
        self.timespace_shape=(args.history_length,*args.img_size) if args.history_length>1 else args.img_size

    def forward(self, x, x_timestamp, y_timestamp):
        # assume input is (B, P ,T, h ,w)
        assert x.shape[2:] == self.timespace_shape
        x_timestamp = x_timestamp.squeeze(1)
        y_timestamp = y_timestamp.squeeze(1)
        if x_timestamp.shape[-1]==4:
            x_timestamp = x_timestamp[...,[0,-1]] # if the time feature is 4 then only the first and last is needed
            y_timestamp = y_timestamp[...,[0,-1]]
        B, P = x.shape[:2]
        direction_shape=[B,-1]+[1]*len(self.timespace_shape)
        year_pos_x = x_timestamp[...,0].reshape(*direction_shape)
        day_pos_x  = x_timestamp[...,1].reshape(*direction_shape)
        x   = torch.cat([x*torch.cos(year_pos_x),
                    x*torch.sin(year_pos_x)*torch.cos(day_pos_x),
                    x*torch.cos(year_pos_x)*torch.sin(day_pos_x)],1) # (B, 3P ,T, h ,w)
        
        x = self.backbone(x)
        x = x.reshape(B,3,P,*self.timespace_shape)
        extra_loss= 0
        if not self.block_target_timestamp:
            year_pos_y = y_timestamp[...,0].reshape(*direction_shape)
            day_pos_y = y_timestamp[...,1].reshape(*direction_shape)
            output_direction = torch.stack([torch.cos(year_pos_y),
                             torch.sin(year_pos_y)*torch.cos(day_pos_y),
                             torch.cos(year_pos_y)*torch.sin(day_pos_y)],1)
            extra_loss = torch.nn.functional.cosine_similarity(x,output_direction,dim=1).mean()
        x = x.norm(dim=1)
        return x, extra_loss
