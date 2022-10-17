import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from utils.tools import get_center_around_indexes

class AdaptiveBatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)' .format(input.dim()))
    def forward(self, input):
        self._check_input_dim(input)
        #!!!below is the only different
        if input.shape[-2]==1:return input

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:exponential_average_factor = 0.0
        else:exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:bn_training = True
        else:bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, in_channels, output_channels, stride,mid_channels=None,upsample=None, norm_layer=AdaptiveBatchNorm2d,relu=nn.ReLU(inplace=True)):
        super().__init__()
        if mid_channels is None:mid_channels=in_channels
        optpad = stride-1
        self.conv1    = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1      = norm_layer(mid_channels)
        self.conv2    = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, bias=False)
        self.bn2      = norm_layer(mid_channels)
        self.conv3    = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=False)
        self.bn3      = norm_layer(output_channels)
        self.relu     = relu
        #self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride   = stride
        self.upsample = upsample
        if self.upsample is None:
            if output_channels!=in_channels or stride >1:
                self.upsample = nn.Sequential(
                        nn.Conv2d(in_channels,
                                           output_channels,
                                           kernel_size=3,
                                           stride=stride,
                                           bias=False) ,
                        norm_layer(output_channels),
                    )
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class NaiveConvModel2D(nn.Module):
    '''
    input is (B, P, patch_range_1,patch_range_2)
    output is (B,P)
    '''
    def __init__(self,img_size=None,patch_range=5,in_chans=20, out_chans=20, **kargs):
        super().__init__()
        self.img_size = img_size
        self.patch_range = 5
        if self.patch_range == 5:
            self.backbone = nn.Sequential(Bottleneck(in_chans,1024,1),
                                          Bottleneck(1024,out_chans,1)
                                         )
            self.mlp = nn.Linear(patch_range**2,1)                            
        else:
            raise NotImplementedError
        self.center_index,self.around_index=get_center_around_indexes(self.patch_range,self.img_size)

    def forward(self, x):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        ''' 
        assert len(x.shape)==4
        input_is_full_image = False
        if x.shape[-2:] == self.img_size:
            input_is_full_image = True
            x = x[...,self.around_index[:,:,0],self.around_index[:,:,1]] # (B,P,W-2,H-2,Patch,Patch)
            x = x.permute(0,2,3,1,4,5)
            B,W,H,P,_,_ = x.shape
            x = x.flatten(0,2) # (B* W-2 * H-2,Patch,Patch)
        x = self.backbone(x).squeeze(-1).squeeze(-1) + self.mlp(x.flatten(-2,-1)).squeeze(-1)
        if input_is_full_image:
            x = x.reshape(B,W,H,P).permute(0,3,1,2)
        return x
        
