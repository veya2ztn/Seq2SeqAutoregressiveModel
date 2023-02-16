import torch,time
import torch.nn as nn
import torch.nn.functional as F
import torch

class BaseModel(nn.Module):

    def set_epoch(self,epoch,epoch_total,**kargs):
        pass
    def set_step(self,step,epoch,**kargs):
        pass
    def get_w_resolution_pad(self,shape):
        w_now   = shape[-2]
        w_should= self.img_size[-2]
        if w_now == w_should:return None
        if w_now > w_should:
            raise NotImplementedError
        if w_now < w_should and not ((w_should - w_now)%2):
            # we only allow symmetry pad
            return (w_should - w_now)//2

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
