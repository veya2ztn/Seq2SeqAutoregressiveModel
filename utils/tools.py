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

def get_center_around_indexes(patch_range,img_shape, h_range=None, w_range=None):
    hlist   = range(patch_range//2, img_shape[-2] - (patch_range//2)) if h_range is None else h_range
    wlist   = range(img_shape[-1]) if w_range is None else w_range
    xes,yes = np.meshgrid(hlist,wlist)
    coor    = np.stack([xes,yes],-1).reshape(-1,2)
    indexes = np.array([np.stack(get_patch_location_index([x,y],img_shape,patch_range)) for x,y in coor] )
    indexes = indexes.reshape(len(wlist),len(hlist),2,patch_range,patch_range).transpose(1,0,2,3,4)
    coor    = coor.reshape(len(wlist),len(hlist),2).transpose(2,1,0)
    return coor, indexes


def get_patch_location_index_3D(center,img_shape,patch_range):
        # we want to get the patch index around center with the patch_range
        # For example, 
        #   (i-1,j-1) (i ,j-1) (i+1,j-1)
        #   (i-1,j ) (i ,j ) (i+1,j )
        #   (i-1,j+1) (i ,j+1) (i+1,j+1)
        # notice our data is on the sphere, this mean the center in H should be in [-boundary+patch_range, boundary-patch_range]
        # and the position in W is perodic.
        assert center[-2] >= patch_range//2
        assert center[-2] <= img_shape[-2] - (patch_range//2)
        assert center[-3] >= patch_range//2
        assert center[-3] <= img_shape[-2] - (patch_range//2)
        delta = [list(range(-(patch_range//2),patch_range//2+1))]*len(center)
        delta = np.meshgrid(*delta)
        pos  = [c+dc for c,dc in zip(center,delta)]
        pos[-1]= pos[-1]%img_shape[-1] # perodic
        pos = np.stack(pos).transpose(0,3,2,1)
        return pos

def get_center_around_indexes_3D(patch_range,img_shape, h_range=None, w_range=None,z_range=None):
    wlist   = range(img_shape[-1]) if w_range is None else w_range
    hlist   = range(patch_range//2, img_shape[-2] - (patch_range//2)) if h_range is None else h_range
    zlist   = range(patch_range//2, img_shape[-3] - (patch_range//2)) if z_range is None else z_range
    zes,yes,xes = np.meshgrid(zlist,hlist,wlist)
    coor    = np.stack([zes,yes,xes],-1).reshape(-1,3)
    indexes = np.array([np.stack(get_patch_location_index_3D([z,y,x],img_shape,patch_range)) for z,y,x in coor] )
    indexes = indexes.reshape(len(wlist),len(hlist),len(zlist),3,patch_range,patch_range,patch_range).transpose(2,1,0,3,4,5,6)
    coor    = coor.reshape(len(wlist),len(hlist),len(zlist),3).transpose(3,2,1,0)
    return coor, indexes

if __name__ == '__main__':
    img_shape = (32,64)
    patch_range = 5
    test_tensor = np.arange(32*64).reshape(32,64)
    coor, indexes = get_center_around_indexes(patch_range,img_shape)
    center_value_array_from_index = test_tensor[coor[0],coor[1]]
    pos_x   = 2
    pos_y   = 4
    center_x,center_y = coor[:,pos_x,pos_y]
    center_value      = test_tensor[center_x,center_y]
    around_x,around_y = indexes[pos_x,pos_y]
    around            = test_tensor[around_x,around_y]
    center_value_from_index = center_value_array_from_index[2,4]
    print(f'''
    the goal of `get_center_around_indexes` and `get_patch_location_index` is to get a 
    - coor[center_tensor] = (2, 28, 64) # if we use (32, 64) earth grid as example,
        - each slot in (28,64) is an pos size like (2,). for example [1,2]
    - indexes[around_tensor] = (28, 64, 2, 5, 5)
        - each slot in (28,64) is an pos array like (2,5,5). 
            the first (1,5,5) is all the x-index and next (1,5,5) is all the y-index
    Here is 28 rather than 32 is because we temperal dont deal with the boundary condition in lattitude.
    And the longitude is obviously peroid.
    for example, if we have a naive increase matrix like 
    ''')
    print(test_tensor)
    print("then the cneter_tensor get by `test_tensor[coor[0],coor[1]]` ")
    print(center_value_array_from_index)
    print(f"if we pick i={pos_x},y={pos_y} block here, then the center is ")

    print(center_value)
    print('and the center value can get from `center_x,center_y = coor[:,pos_x,pos_y];center_value      = test_tensor[center_x,center_y]`')
    print(around)
    print("its is obviously got from the subarray by center_value_array_from_index[2,4]")
    print(center_value_from_index)


    