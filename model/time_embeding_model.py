import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .afnonet import BaseModel


class Sphere_Model(BaseModel):
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone =  backbone
        self.block_target_timestamp = args.block_target_timestamp
        self.timespace_shape=(args.history_length,*args.img_size) 
        self.time_length = args.history_length
        self.space_shape = args.img_size
        self.pred_len = args.pred_len

    
    def get_direction_from_time_stamp(self,x_timestamp):
        x_timestamp = x_timestamp*np.pi # the input is in [-1,1]
        if x_timestamp.shape[-1]==4:x_timestamp = x_timestamp[...,[0,-1]] # if the time feature is 4 then only the first and last is needed
        year_pos_x = x_timestamp[...,0]
        day_pos_x  = x_timestamp[...,1]
        x_direction = torch.stack([torch.cos(year_pos_x),
                       torch.sin(year_pos_x)*torch.cos(day_pos_x),
                       torch.cos(year_pos_x)*torch.sin(day_pos_x)],1) # (B, 3 ,T)
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
        assert x.shape[2:]      == self.timespace_shape
        assert x_timestamp.shape[1] == self.time_length
        B, P = x.shape[:2]
        
        x_direction = self.get_direction_from_time_stamp(x_timestamp)
        y_direction = self.get_direction_from_time_stamp(y_timestamp)
        x = torch.einsum('bpt...,bdt->bdpt...',x,x_direction).flatten(1,2)
        x = self.backbone(x)
        x = x.reshape(B, 3, P, self.pred_len,*self.space_shape)
        
        extra_loss= 0
        if not self.block_target_timestamp:
            Pshape = [B,3]+[1]*(len(x.shape)-2)
            y_direction = y_direction.reshape(*Pshape)
            extra_loss = 1 - torch.nn.functional.cosine_similarity(x,y_direction,dim=1).mean()
        x = x.norm(dim=1)
        return x, extra_loss

class Time_Projection_Model(Sphere_Model):
    extra_loss_coef =  1
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

    def set_epoch(self,epoch,epoch_total):
        self.extra_loss_coef = (np.exp(1) - np.exp(epoch/epoch_total))/(np.exp(1) - 1)

    def forward(self, x, x_timestamp, y_timestamp):
        # assume input is (B, P ,T, h ,w)
        assert self.pred_len == 1
        assert x.shape[2:]      == self.timespace_shape
        assert x_timestamp.shape[1] == self.time_length
        B, P = x.shape[:2]
        
        x_direction = self.get_direction_from_time_stamp(x_timestamp)
        y_direction = self.get_direction_from_time_stamp(y_timestamp)
        y = torch.einsum('bpt...,bdt->bdpt...',x,x_direction).flatten(1,2)
        y = self.backbone(y)
        y = y.reshape(B, 3, P, *self.space_shape) 
        # should be (B, 3, P,self.pred_len,  *self.space_shape). omit self.pred_len since it is set 1
        
        self_projection = torch.einsum('bdp...,bdt->bpt...',y, x_direction)
        extra_loss   = F.mse_loss(self_projection,x)  
        
        target_projection = torch.einsum('bdp...,bdt->bpt...',y, y_direction)
        return target_projection, self.extra_loss_coef*extra_loss
