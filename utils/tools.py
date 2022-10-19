import numpy as np
import torch
import os

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return param_sum, buffer_sum, all_size


def load_model(model, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False,loc = 'cuda:0'):

    start_epoch, start_step = 0, 0
    min_loss = np.inf
    if os.path.exists(path) and path != "":
        print(f"loading model from {path}...........")
        ckpt = torch.load(path, map_location='cpu')

        if only_model:
            
            model.load_state_dict(ckpt['model'])
            print("loading model weight success...........")
        else:
            model.load_state_dict(ckpt['model'])
            print("loading model weight success...........")
            optimizer.load_state_dict(ckpt['optimizer'])
            print("loading optimizer weight success...........")
            if lr_scheduler is not None:lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            print("loading lr_scheduler weight success...........")
            loss_scaler.load_state_dict(ckpt['loss_scaler'])
            print("loading loss_scaler weight success...........")
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["min_loss"]
        print("loading model success...........")
    else:
        print("dont find path, please check, we pass........")
    return start_epoch, start_step, min_loss


def save_model(model, epoch=0, step=0, optimizer=None, lr_scheduler=None, loss_scaler=None, min_loss=0, path=None, only_model=False):

    if only_model:
        states = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        }
    else:
        states = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'loss_scaler': loss_scaler.state_dict(),
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss
        }

    torch.save(states, path)

def get_patch_location_index(center,img_shape,patch_range):
        # we want to get the patch index around center with the self.patch_range
        # For example, 
        #   (i-1,j-1) (i ,j-1) (i+1,j-1)
        #   (i-1,j ) (i ,j ) (i+1,j )
        #   (i-1,j+1) (i ,j+1) (i+1,j+1)
        # notice our data is on the sphere, this mean the center in H should be in [-boundary+patch_range, boundary-patch_range]
        # and the position in W is perodic.
        assert center[-2] >= patch_range//2
        assert center[-2] <= img_shape[-2] - (patch_range//2)
        delta = [list(range(-(patch_range//2),patch_range//2+1))]*len(center)
        delta = np.meshgrid(*delta)
        pos  = [c+dc for c,dc in zip(center,delta)]
        pos[-1]= pos[-1]%img_shape[-1] # perodic
        pos = np.stack(pos).transpose(0,2,1)
        return pos

def get_center_around_indexes(patch_range,img_shape):
    hlist = range(patch_range//2, img_shape[-2] - (patch_range//2))
    wlist = range(img_shape[-1])
    xes,yes = np.meshgrid(hlist,wlist)
    coor   = np.stack([xes,yes],-1).reshape(-1,2)
    indexes = np.array([np.stack(get_patch_location_index([x,y],img_shape,patch_range)) for x,y in coor] )
    indexes = indexes.reshape(len(wlist),len(hlist),2,patch_range,patch_range).transpose(1,0,2,3,4)
    coor    = coor.reshape(len(wlist),len(hlist),2).transpose(2,1,0)
    return coor, indexes