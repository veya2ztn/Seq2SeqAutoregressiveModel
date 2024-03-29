import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from utils.tools import get_center_around_indexes
# from vit_pytorch import ViT
# from vit_pytorch.vit import repeat
# from .custom_transformer import ViT3DTransformer, ViT3DFlowformer
# from .FEDformer import FEDformer,Informer
from einops import rearrange

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
            self.backbone = nn.Sequential(Bottleneck(in_chans,1024,1),Bottleneck(1024,out_chans,1))
            self.mlp = nn.Linear(self.patch_range**2, 1)
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
            x = x[...,self.around_index[:,:,0],self.around_index[:,:,1]] # (B,P,W-4,H,Patch,Patch)
            x = x.permute(0,2,3,1,4,5)
            B,W,H,P,_,_ = x.shape
            x = x.flatten(0,2) # (B* W-4 * H,Patch,Patch)
        x = self.backbone(x).squeeze(-1).squeeze(-1) + self.mlp(x.flatten(-2,-1)).squeeze(-1)
        if input_is_full_image: 
            x = x.reshape(B,W,H,P).permute(0,3,1,2)
        return x

class AutoPatchModel2D:
    center_around_index_table = {}
    # def __init__(self, config):
    #     super().__init__()
    #     #assert img_size is not None
    #     #print(f"for 2D patch model, the img_size will be force set (32,64)")
    #     self.img_size    = config.img_size
    #     self.patch_range = config.patch_range
    #     self.center_index,self.around_index=get_center_around_indexes(self.patch_range,self.img_size)
    #     self.center_index_pool={}
    #     self.around_index_pool={}
    #     img_size=tuple(img_size)
    #     self.center_index_pool[img_size]=self.center_index
    #     self.around_index_pool[img_size]=self.around_index

    def create_center_around_index(self,img_shape):
        patch_range = self.patch_range if isinstance(self.patch_range,(list,tuple)) else (self.patch_range,self.patch_range)
        img_shape   = tuple(img_shape)
        h_range     = range(patch_range[-2]//2, img_shape[-2] - (patch_range[-2]//2))
        if img_shape[-1] != self.img_size[-1]: # if is not 64, then we disable peroid boundary
            w_range = range(patch_range[-1]//2, img_shape[-1] - (patch_range[-1]//2))
        else:
            w_range = range(img_shape[-1]) 
        center_index,around_index=get_center_around_indexes(patch_range,img_shape, h_range, w_range)
        self.center_index_pool[img_shape]=center_index
        self.around_index_pool[img_shape]=around_index
        
    def center_index_depend_input(self, img_shape):
        if img_shape not in self.center_index_pool:
            self.create_center_around_index(img_shape)
        return self.center_index_pool[img_shape]

    def around_index_depend_input(self, img_shape):
        if img_shape not in self.around_index_pool:
            self.create_center_around_index(img_shape)
        return self.around_index_pool[img_shape]

    def get_center_index_depend_on(self, tgt_shape, img_shape):
        if img_shape not in self.center_around_index_table:
            self.center_around_index_table[img_shape]={}
        if tgt_shape not in self.center_around_index_table[img_shape]:
            h_delta = img_shape[0] - tgt_shape[0]
            h_range = range(h_delta//2, img_shape[0] - h_delta//2)
            assert len(h_range) == tgt_shape[0]
            w_delta = img_shape[1] - tgt_shape[1]
            w_range = range(w_delta//2, img_shape[1] - w_delta//2)
            assert len(w_range) == tgt_shape[1]
            self.center_around_index_table[img_shape][tgt_shape]=get_center_around_indexes(self.patch_range,img_shape, h_range=h_range, w_range=w_range)
        return self.center_around_index_table[img_shape][tgt_shape]

    @staticmethod
    def good_tuple(x, num):
        if isinstance(x,(list,tuple)):
            return tuple(x)
        else:
            return tuple([x]*num)

    def image_to_patches(self, x):
        if len(x.shape) == 3:x=x.unsqueeze(0)
        assert len(x.shape)==4
        self.input_is_full_image = False
        good_input_shape = self.good_tuple(self.patch_range,len(self.img_size))
        now_input_shape  = tuple(x.shape[-2:])
        if  now_input_shape!= good_input_shape:
            self.input_is_full_image = True
            around_index = self.around_index_depend_input(now_input_shape)
            x = x[...,around_index[:,:,0],around_index[:,:,1]] # (B,P,W-4,H,Patch,Patch)
            x = x.permute(0,2,3,1,4,5)
            B,W,H,P,_,_ = x.shape
            self.input_shape_tmp=(B,W,H,P)
            x = x.flatten(0,2) # (B* W-4 * H,P, Patch,Patch)
        now_input_shape  = tuple(x.shape[-2:])
        assert now_input_shape == good_input_shape, f"input shape is {now_input_shape}, but should be {good_input_shape}"
        return x

    def patches_to_image(self,x):
        if self.input_is_full_image: 
            B,W,H,P = self.input_shape_tmp
            x = x.reshape(B,W,H,P).permute(0,3,1,2)
        return x

class AutoPatchOverLapModel2D(AutoPatchModel2D):
    counting_matrix = None
    counting_matrixes = {}
    yeses = {}
    peses = {}
    
    # def patches_to_image(self,x):
    #     """
    #     Should split to two types:
    #     1. if the output tensor is  (32,64), then it has global period symmetry
    #     2. if the output tensor is bigger patch like (8,8), the it means in finetune phase and don't have peroid symmetry. 
    #     """
    #     if self.input_is_full_image: 
    #         assert self.patch_range%2 ==1
    #         B,W,H,P  = self.input_shape_tmp
    #         L     =  W + 2*(self.patch_range//2) #32
    #         # counting_matrix[0]*=5
    #         # counting_matrix[1]*=10
    #         # counting_matrix[2]*=15
    #         # counting_matrix[3]*=20
    #         # counting_matrix[4:W]*=25
    #         # counting_matrix[W]*=20
    #         # counting_matrix[W+1]*=15
    #         # counting_matrix[W+2]*=10
    #         # counting_matrix[W+3]*=5
    #         # w_idx   = np.arange(0,L)
    #         # self.wes  = np.stack([w_idx, w_idx+1,w_idx+2, w_idx-1, w_idx-2],1)%L
    #         # x_idx   = np.arange(H)
    #         # self.xes = np.stack([x_idx, x_idx+1,x_idx+2, x_idx-1, x_idx-2],1)%H
    #         # self.yes = np.array([[2,  1,  0,  3,  4]])
    #         if self.counting_matrix is None:
    #             #counting_matrix = torch.ones(L,H)
    #             #for i in range(self.patch_range-1):
    #             #    counting_matrix[i]*= self.patch_range*(i+1)
    #             #    counting_matrix[W+i]*= self.patch_range*(self.patch_range-i-1)
    #             #counting_matrix[self.patch_range-1:W]*= self.patch_range*self.patch_range
    #             #self.counting_matrix = counting_matrix.unsqueeze(0).unsqueeze(0) #(1,1,32,64)
    #             self.counting_matrix = torch.nn.functional.conv2d(torch.ones(1,1,W,H + 2*(self.patch_range//2)),torch.ones(1,1,self.patch_range,self.patch_range),padding=(2*(self.patch_range//2),0),stride=1)
            
    #             Delta   = list(range(self.patch_range)) #[0,1,2,3,4]
    #             self.yes  = np.array([Delta])
    #             Delta   =  -np.array(Delta) + self.patch_range//2 #-->[2,1,0,-1,-2]
                
    #             w_idx   = np.arange(L)
    #             self.wes  = np.stack([w_idx+d for d in Delta],1)%L
    #             x_idx   = np.arange(H)
    #             self.xes = np.stack([x_idx+d for d in Delta],1)%H
    #             #print(self.yes)
    #             #print(Delta)
    #         x = x.reshape(B,W,H,P,self.patch_range,self.patch_range)
    #         x = torch.nn.functional.pad(x,(0,0, 0,0, 0,0, 0,0, self.patch_range//2 , self.patch_range//2 )) # only extend W    
    #         x     = x[:, self.wes, :,:,self.yes,:].sum(1) #(B, W, H, P, PS,PS) --> (W, B, H, P, PS)
    #         x     = x[:, :, self.xes,:,self.yes].sum(1)  #(W, B, H, P, PS) --> (H, W, B, P)   
    #         x     = x.permute(2,3,1,0)#(B,P, W,H)   
    #         self.counting_matrix =self.counting_matrix.to(x.device)
    #         x     = x/self.counting_matrix #(B,P, W,H)  / (1,1 , W,H)
    #     return x
    
    def patches_to_image(self,x):
        if self.input_is_full_image: 
            B,X0,X1,P  = self.input_shape_tmp
            if X1 == self.img_size[-1]:
                x = self.patches_to_image_full(x)
            else:
                x = self.patches_to_image_adapt(x)
        return x
    
    def patches_to_image_adapt(self,x):
        # the goal is to recovery (LargeBatch,LargeBatch) from (LargeBatch - 2,LargeBatch -2, Patch_Size, Patch_Size)
        patch_range = self.patch_range if isinstance(self.patch_range,(list,tuple)) else (self.patch_range,self.patch_range)
        assert (patch_range[0]%2 ==1) and (patch_range[1]%2 ==1) 
        B,X0,X1,P  = self.input_shape_tmp
        # assume X2 is peroidic
        Y0 = X0+ 2*(patch_range[0]//2) #14
        Y1 = X1+ 2*(patch_range[1]//2) #32
        if (X0,X1) not in self.counting_matrixes:
            self.counting_matrixes[X0,X1] = torch.nn.functional.conv2d(
                torch.ones(1,1,X0,X1),
                torch.ones(1,1,*patch_range),
                padding=(2*(patch_range[0]//2),2*(patch_range[1]//2)),
                stride=1)
            self.yeses[X0,X1] = {}
            self.peses[X0,X1] = {}
            for idx, (p, Y) in enumerate(zip(patch_range,(Y0,Y1))):
                pes  = np.arange(p)
                Delta= np.arange(p//2,-p//2,-1)
                yes  = np.arange(Y)
                yes  = np.stack([yes+d for d in Delta],1)%Y
                self.yeses[X0,X1][idx]=yes
                self.peses[X0,X1][idx]=pes

        counting_matrix =self.counting_matrixes[X0,X1].to(x.device)

        yes = self.yeses[X0,X1]
        pes = self.peses[X0,X1]
        x = x.reshape(B,X0,X1,P,*patch_range)
        x = torch.nn.functional.pad(x,(
                0,0,   0,0,   0,0, 
                patch_range[-1]//2 , patch_range[-1]//2,
                patch_range[-2]//2 , patch_range[-2]//2,
            )) #(B, X0, X1, P, p0,p1) --> (B, Y0, Y1, P, p0,p1)

        x = x[:, yes[0],      :,      :, pes[0],:].sum(1) # (B, Y0, Y1, P, p0,p1)--> (Y0, B, Y1, P, p1)
        x = x[:,      :, yes[1],      :, pes[1], ].sum(1) # (Y0, B, Y1, P, p1)--> (Y1, Y0, B, P)
        x = x.permute(2,3,1,0)#(B,P, W,H)   
        x = x/counting_matrix #(B,P, W,H)  / (1,1 , W,H)
        return x

    def patches_to_image_full(self,x):
        # the goal is to recovery (LargeBatch,LargeBatch) from (LargeBatch - 2,LargeBatch -2, Patch_Size, Patch_Size)
        patch_range = self.patch_range if isinstance(self.patch_range,(list,tuple)) else (self.patch_range,self.patch_range)
        assert (patch_range[0]%2 ==1) and (patch_range[1]%2 ==1) 
        B,X0,X1,P  = self.input_shape_tmp
        # assume X2 is peroidic
        Y0 = X0 + 2*(patch_range[0]//2) #14
        Y1 = X1  #32
        assert Y1 == self.img_size[-1]
        if (Y0,Y1) not in self.counting_matrixes:
            self.counting_matrixes[Y0,Y1] = torch.nn.functional.conv2d(
                torch.ones(1,1,X0,X1+2*(patch_range[1]//2)),
                torch.ones(1,1,*patch_range),
                padding=(2*(patch_range[0]//2),0),
                stride=1)

            self.yeses[Y0,Y1] = {}
            self.peses[Y0,Y1] = {}
            for idx, (p, Y) in enumerate(zip(patch_range,(Y0,Y1))):
                pes  = np.arange(p)
                Delta= np.arange(p//2,-p//2,-1)
                yes  = np.arange(Y)
                yes  = np.stack([yes+d for d in Delta],1)%Y
                self.yeses[Y0,Y1][idx]=yes
                self.peses[Y0,Y1][idx]=pes

        counting_matrix =self.counting_matrixes[Y0,Y1].to(x.device)

        yes = self.yeses[Y0,Y1]
        pes = self.peses[Y0,Y1]
        x = x.reshape(B,X0,X1,-1,*patch_range)
        x = torch.nn.functional.pad(x,(
                0,0,   0,0,   0,0,  0, 0,
                patch_range[-2]//2 , patch_range[-2]//2,
            )) #(B, X0, X1, P, p0,p1) --> (B, Y0, Y1, P, p0,p1)

        x = x[:, yes[0],      :,      :, pes[0],:].sum(1) # (B, Y0, Y1, P, p0,p1)--> (Y0, B, Y1, P, p1)
        x = x[:,      :, yes[1],      :, pes[1], ].sum(1) # (Y0, B, Y1, P, p1)--> (Y1, Y0, B, P)
        x = x.permute(2,3,1,0)#(B,P, W,H)   
        x = x/counting_matrix #(B,P, W,H)  / (1,1 , W,H)
        return x

    def patches_to_image_slow(self,x):
        if self.input_is_full_image: 
            B,W,H,P = self.input_shape_tmp
            # (28, 32)
            x = x.reshape(B,W,H,P,self.patch_range,self.patch_range)
            assert self.patch_range==5
            assert not self.training
            
            x_idx = np.arange(H)
            xes = np.stack([x_idx, x_idx+1,x_idx+2, x_idx-1, x_idx-2],1)%H
            yes = np.array([[2,  1,  0,  3,  4]])
            lines = []
            end = W + 4 
            for line_id in range(end): #(0 --> 32)
                line = 0
                if line_id < 4:
                    for w_id in range(line_id+1):
                        line += x[:, w_id, xes,:, line_id - w_id,yes].mean(1) #(H,B,P)
                    line = line/(line_id + 1)
                    line = line.permute(1,2,0).unsqueeze(2)#(B,P,1,H)      
                elif line_id > end - 5:
                    for w_id in range(line_id - end, 0): #(-3,-2,-1)
                        line += x[:, w_id, xes,:, line_id - end -1 - w_id,yes].mean(1)#(H,B,P)       
                    line = line/(end - line_id)
                    line = line.permute(1,2,0).unsqueeze(2)#(B,P,1,H)       
                elif line_id == 4:
                    w_idx = np.arange(2,W-2)
                    wes   = np.stack([w_idx, w_idx+1,w_idx+2, w_idx-1, w_idx-2],1)
                    line  = x[:, wes, :,:,yes,:].mean(1) #(4,B, H, P, PS)
                    line  = line[:, :, xes,:,yes].mean(1)#(H,B,P)   
                    line  = line.permute(2,3,1,0)
                else:
                    continue
                lines.append(line)
            x = torch.cat(lines,2)

        return x

class POverLapTimePosBias2D(AutoPatchOverLapModel2D):
    '''
    input is (B, P, patch_range_1,patch_range_2) tensor
             (B, 4) time_stamp
             (B, 2, patch_range_1,patch_range_2) pos_stamp
    output is (B,P, patch_range_1,patch_range_2)
    '''
    global_position_feature = None

    def get_direction_from_stamp(self,stamp):
        w_pos_x    = stamp[:,0]/32*np.pi
        h_pos_x    = stamp[:,1]/64*2*np.pi
        x_direction = torch.stack([torch.cos(w_pos_x),
                                   torch.sin(w_pos_x)*torch.cos(h_pos_x),
                                   torch.cos(w_pos_x)*torch.sin(h_pos_x)],1) # (B, 3 ,patch_range_1,patch_range_2)
        return x_direction

    def image_to_patches(self, x,time_stamp , pos_stamp):
        assert len(x.shape)==4
        self.input_is_full_image = False
        good_input_shape = (self.patch_range,self.patch_range)
        now_input_shape  = tuple(x.shape[-2:])
        if  now_input_shape!= good_input_shape:
            self.input_is_full_image = True
            around_index = self.around_index_depend_input(now_input_shape)
            if self.global_position_feature is None:
                grid = torch.Tensor(np.stack(np.meshgrid(np.arange(self.img_size[0]),np.arange(self.img_size[1]))).transpose(0,2,1)) #(2, 32, 64)
                self.global_position_feature = self.get_direction_from_stamp(grid[None]) #(1, 3, W, H)
            if torch.all(pos_stamp==-1):
                pos_feature = self.global_position_feature.repeat(x.size(0),1,1,1).to(x.device) #(B, 3, W, H)
            else:
                pos_feature = self.get_direction_from_stamp(pos_stamp)
            B, T = time_stamp.shape
            time_feature= time_stamp.reshape(B,T,1,1).repeat(1,1,*x.shape[-2:])#(B, 4, W, H)
            B,P,_,_ = x.shape
            x = torch.cat([x,pos_feature,time_feature],1) #( B, 77, W, H)
            x = x[...,around_index[:,:,0],around_index[:,:,1]] # (B,P,W-4,H,Patch,Patch)
            x = x.permute(0,2,3,1,4,5)
            _,W,H,_,_,_ = x.shape
            self.input_shape_tmp=(B,W,H,P)
            x = x.flatten(0,2) # (B* W-4 * H,P, Patch,Patch)
        else:
            pos_feature = self.get_direction_from_stamp(pos_stamp)
            B, T = time_stamp.shape
            time_feature= time_stamp.reshape(B,T,1,1).repeat(1,1,self.patch_range,self.patch_range)
            x = torch.cat([x,pos_feature,time_feature],1) #( B, 77, Patch, Patch)
        now_input_shape  = tuple(x.shape[-2:])
        assert now_input_shape == good_input_shape
        
        return x

class PatchOverLapWrapper(AutoPatchOverLapModel2D):
    '''
    input is (B, P, patch_range_1,patch_range_2)
    output is (B,P, patch_range_1,patch_range_2)
    '''

    def __init__(self, args, backbone):
        patch_range = 5 if args is None else args.patch_range
        super().__init__((32,64),patch_range)
        self.backbone    = backbone
        self.monitor     = True
        self.img_size    = (32, 64)
        self.patch_range = patch_range
        
    def forward(self, x):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        x = self.image_to_patches(x)
        x = self.backbone(x)
        x = self.patches_to_image(x)
        return x




class POverLapTimePosBiasWrapper(POverLapTimePosBias2D):
    def __init__(self, args, backbone):
        patch_range = 5 if args is None else args.patch_range
        super().__init__((32,64),patch_range)
        self.backbone = backbone
        self.monitor = True
        self.img_size = (32, 64)
        self.patch_range = patch_range
    def forward(self, x, time_stamp=None, pos_stamp=None ):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        if len(x.shape)==2 and time_stamp is None and pos_stamp is None:
            xshape = (70,self.patch_range,self.patch_range)
            tshape = (4,)
            pshape = (2,self.patch_range,self.patch_range)
            a = np.prod(xshape)
            b = np.prod(tshape)
            c = np.prod(pshape)    
            time_stamp = x[:,a:a+b].reshape(-1,*tshape)
            pos_stamp = x[:,a+b:a+b+c].reshape(-1,*pshape)
            x = x[:,0:a].reshape(-1,*xshape)
        x = self.image_to_patches(x,time_stamp,pos_stamp)
        x = self.backbone(x)
        #print(x.shape)
        x = self.patches_to_image(x)
        return x

class POverLapTimePosFEDformer(POverLapTimePosBias2D):
    '''
    input is (B, P, patch_range_1,patch_range_2) tensor
             (B, 2, patch_range_1,patch_range_2) pos_stamp
             (B, 4) start_time_stamp
             (B, 4) end_time_stamp
    output is (B,P, patch_range_1,patch_range_2)
    '''
    def __init__(self, img_size=None,patch_range=5,in_chans=73, out_chans=70,**kargs):
        super().__init__(img_size,patch_range)
        self.in_chans=in_chans
        self.out_chans=out_chans
        self.patch_range = patch_range
        self._build_backbone(img_size=img_size,patch_range=patch_range,in_chans=in_chans, out_chans=out_chans,**kargs)

    def _build_backbone(self,img_size=None,patch_range=5,in_chans=73, out_chans=70,**kargs):
        self.backbone = FEDformer(img_size=(patch_range,patch_range), in_chans=in_chans,out_chans=out_chans, 
                                  embed_dim=kargs.get('embed_dim',748), 
                                  modes=kargs.get('modes',(4,4,6)), 
                                  n_heads=kargs.get('n_heads', 24),
                                  depth=kargs.get('model_depth',4),
                                  history_length=kargs.get('history_length',16),
                                  pred_len=kargs.get('pred_len',4),
                                  label_len=kargs.get('label_len',4),share_memory=kargs.get('share_memory',1))

    def image_to_patches(self, x,pos_stamp,start_time_stamp,end_time_stamp):
        assert len(x.shape)==5 #(B, T, P, W, H)
        Batch_size, Time_length = x.shape[:2]
        x = x.flatten(0,1)

        self.input_is_full_image = False
        good_input_shape = (self.patch_range,self.patch_range)
        now_input_shape  = tuple(x.shape[-2:])
        if  now_input_shape!= good_input_shape:
            self.input_is_full_image = True
            around_index = self.around_index_depend_input(now_input_shape)
            if self.global_position_feature is None:
                grid = torch.Tensor(np.stack(np.meshgrid(np.arange(self.img_size[0]),np.arange(self.img_size[1]))).transpose(0,2,1)) #(2, 32, 64)
                self.global_position_feature = self.get_direction_from_stamp(grid[None]) #(1, 3, W, H)
            pos_feature  = self.global_position_feature.repeat(x.size(0),1,1,1).to(x.device) #(B, 3, W, H)
            B,P,_,_ = x.shape
            
            x = torch.cat([x,pos_feature],1) #( B, 77, W, H)
            x = x[...,around_index[:,:,0],around_index[:,:,1]] # (B,P,W-4,H,Patch,Patch)
            x = x.permute(0,2,3,1,4,5)
            _,W,H,PP,_,_ = x.shape
            output_batch = np.prod(end_time_stamp.shape[:2])
            self.input_shape_tmp=(output_batch,W,H,P)
            start_time_stamp = start_time_stamp.unsqueeze(1).unsqueeze(1).repeat(1,W,H,1,1).flatten(0,2)
            end_time_stamp   = end_time_stamp.unsqueeze(1).unsqueeze(1).repeat(1,W,H,1,1).flatten(0,2)
            
            x = x.flatten(0,2) # (B* W-4 * H,P, Patch,Patch)
        else:
            pos_stamp = pos_stamp.flatten(0,1)
            pos_feature = self.get_direction_from_stamp(pos_stamp) # ( B, T, 73, Patch, Patch)
            x = torch.cat([x,pos_feature],1) #( B, T, 73, Patch, Patch)
        now_input_shape  = tuple(x.shape[-2:])
        assert now_input_shape == good_input_shape
        x = x.reshape(-1, Time_length, *x.shape[1:])
        return x,start_time_stamp,end_time_stamp

    def forward(self, x, pos_stamp, start_time_stamp, end_time_stamp ):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        #print(x.shape)
        shape = x.shape
        x,start_time_stamp,end_time_stamp = self.image_to_patches(x,pos_stamp,start_time_stamp,end_time_stamp)
        x = rearrange(x,'b t c w h -> b w h t c')
        TheMinBatch=1024
        if len(x) > TheMinBatch: #(B, LargeP, P, p0, p1, p2)
            assert not self.training
            x = torch.cat([self.backbone(t1,t2,t3) for t1,t2,t3 in zip(torch.split(x,TheMinBatch),torch.split(start_time_stamp,TheMinBatch),torch.split(end_time_stamp,TheMinBatch))])
        else:   
            x = self.backbone(x,start_time_stamp, end_time_stamp)
        
        x = rearrange(x,'b w h t c -> b t c w h')
        x = x.flatten(0,1)
        x = self.patches_to_image(x)
        x = x.reshape(shape[0],-1,*shape[2:])
        return x

class POverLapTimePosInformer(POverLapTimePosFEDformer):
    '''
    input is (B, P, patch_range_1,patch_range_2) tensor
             (B, 2, patch_range_1,patch_range_2) pos_stamp
             (B, 4) start_time_stamp
             (B, 4) end_time_stamp
    output is (B,P, patch_range_1,patch_range_2)
    '''
    def _build_backbone(self,img_size=None,patch_range=5,in_chans=73, out_chans=70,**kargs):
        self.backbone = Informer(img_size=(patch_range,patch_range), in_chans=in_chans,out_chans=out_chans, 
                                  embed_dim=kargs.get('embed_dim',748), 
                                  modes=kargs.get('modes',(4,4,6)), 
                                  n_heads=kargs.get('n_heads', 24),
                                  depth=kargs.get('model_depth',4),
                                  history_length=kargs.get('history_length',16),
                                  pred_len=kargs.get('pred_len',4),
                                  label_len=kargs.get('label_len',4))


class LargeMLP(AutoPatchModel2D):
    '''
    input is (B, P, patch_range_1,patch_range_2)
    output is (B,P)
    '''

    def __init__(self, img_size=None, patch_range=5, in_chans=20, out_chans=20, p=0.1, **kargs):
        super().__init__(img_size, patch_range)
        if self.patch_range == 5:
            cl = [5*5*in_chans, 5*5*100, 5*5*100, 5*5*100, 5*5*100, 5*5*100, 5*5*70, 5*5*70, 5*5*70, out_chans]
            nnlist = []
            for i in range(len(cl)-2):
                nnlist += [nn.Linear(cl[i], cl[i+1]),nn.Dropout(p=p), nn.Tanh()]
            nnlist += [nn.Linear(cl[-2], cl[-1])]
            self.backbone = nn.Sequential(*nnlist)
        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        x = self.image_to_patches(x)
        x = self.backbone(x.flatten(-3, -1))  # (B* W-4 * H,P)
        x = self.patches_to_image(x)
        return x

class SimpleViT(AutoPatchModel2D):
    def __init__(self,img_size=None,patch_range=5,in_chans=20, out_chans=20,p=0.1,**kargs):
        super().__init__(img_size,patch_range)
        self.backbone = ViT(image_size = (patch_range,patch_range),
                    patch_size = 1,
                    num_classes = 70,
                    dim = 1024,
                    depth = 7,
                    heads = 16,
                    mlp_dim = 768,
                    channels= 70,
                    dropout = 0.1,
                    emb_dropout = 0.1
            )

    def forward(self, x):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        ''' 
        x = self.image_to_patches(x)
        x = self.backbone(x)
        x = self.patches_to_image(x)
        return x

class OverLapLargeViT(POverLapTimePosBiasWrapper):
    def __init__(self,img_size=None,patch_range=5,in_chans=77, out_chans=70,p=0.1,**kargs):
        super().__init__(None,None)
        self.in_chans=in_chans
        self.out_chans=out_chans
        self.backbone = ViT(image_size = (patch_range,patch_range),
                    patch_size = 1,
                    num_classes = self.out_chans,
                    dim = 2024,
                    depth = 16,
                    heads = 6,
                    mlp_dim = 768,
                    channels= self.in_chans,
                    dropout = 0.1,
                    emb_dropout = 0.1
            )

    def forward(self, x, time_stamp, pos_stamp ):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        #print(x.shape)
        x = self.image_to_patches(x,time_stamp,pos_stamp)
        x = self.backbone.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.backbone.cls_token, '1 1 d -> b 1 d', b = b)
        x  = torch.cat((cls_tokens, x), dim=1)
        x += self.backbone.pos_embedding[:, :(n + 1)]
        x  = self.backbone.dropout(x)
        x  = self.backbone.transformer(x)[:,:-1] #(B, 25, dim)
        x  = self.backbone.mlp_head(x)#(B, 25, 70)
        x  = x.permute(0,2,1).reshape(-1,self.out_chans,self.patch_range,self.patch_range) #(B,70,5,5)
        x  = self.patches_to_image(x)
        return x

class PatchWrapper(AutoPatchModel2D):
    '''
    input is (B, P, patch_range_1,patch_range_2)
    output is (B,P)
    '''

    def __init__(self, args, backbone):
        super().__init__(args.img_size,5)
        self.backbone = backbone
        self.monitor = True
        self.img_size = (32, 64)
        self.patch_range = 5
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(args.output_channel*self.patch_range**2, args.output_channel))

    def forward(self, x):
        '''
        The input either (B,P,patch_range,patch_range) or (B,P,w,h)
        The output then is  (B,P) or (B,P,w-patch_range//2,h-patch_range//2)
        '''
        x = self.image_to_patches(x)
        x = self.backbone(x)
        x = self.mlp(x.reshape(x.size(0), -1))
        x = self.patches_to_image(x)
        return x
