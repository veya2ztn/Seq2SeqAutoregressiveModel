import numpy as np
import torch
import os
import math




def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


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

def load_model(model, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False, strict=True):

    start_epoch, start_step = 0, 0
    min_loss = np.inf
    if path != "":
        assert os.path.exists(path) 
        print(f"loading model from {path}...........")
        ckpt = torch.load(path, map_location='cpu')

        if only_model:

            model_state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            if "loragrashcastdglsym" in model_state_dict:
                model_state_dict = model_state_dict["loragrashcastdglsym"]
            if "lgnet" in model_state_dict:
                model_state_dict = model_state_dict["lgnet"]
            
            old_state_dict = model.state_dict()
            model_keys = list(old_state_dict.keys())
            new_state_dict = {}
            for key, val in model_state_dict.items():
                if '.expand' in key or '.repeat' in key or '.scale' in key:
                    continue
                if "module." in key and np.all(['module.' not in k for k in model_keys]):
                    key = key.replace("module.", "")
                if "_orig_mod." in key and np.all(['_orig_mod.' not in k for k in model_keys]):
                    key = key.replace("_orig_mod.", "")
                if "backbone.net." in key and np.all(['backbone.net.' not in k for k in model_keys]):
                    key = key.replace("backbone.net.", "net.")  
                if "backbone.backbone.net." in key and np.all(['backbone.backbone.net.' not in k for k in model_keys]):
                    key = key.replace(
                        "backbone.backbone.net.", "backbone.net.")
                if "backbone.net." not in key and np.any(['backbone.net.' in k for k in model_keys]):
                    key = key.replace("net.", "backbone.net.")
                if "backbone.backbone.net" not in key and np.any(['backbone.backbone.net.' in k for k in model_keys]):
                    key = key.replace(
                        "backbone.net.", "backbone.backbone.net.")

                new_state_dict[key] = val

            # for key in model_keys:
            #     if '.expand' in key and key not in new_state_dict:new_state_dict[key] = old_state_dict[key]
            #     if '.repeat' in key and key not in new_state_dict:new_state_dict[key] = old_state_dict[key]
            #     if '.scale'  in key and key not in new_state_dict:new_state_dict[key] = old_state_dict[key]
            model_state_dict = new_state_dict

            if 'max_logvar' in model_state_dict:
                del model_state_dict['max_logvar']
            if 'min_logvar' in model_state_dict:
                del model_state_dict['min_logvar']
            model.load_state_dict(model_state_dict, strict=strict)
            print("loading model weight success...........")
        else:
            model_state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            model.load_state_dict(model_state_dict)
            print("loading model weight success...........")
            optimizer.load_state_dict(ckpt['optimizer'])
            print("loading optimizer weight success...........")
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                print("loading lr_scheduler weight success...........")
            else:
                print("loading lr_scheduler weight fail...........")

            loss_scaler.load_state_dict(ckpt['loss_scaler'])
            print("loading loss_scaler weight success...........")
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["best_valid_loss"]
        print("loading model success...........")
    else:
        print("dont find path, please check, we pass........")
    return start_epoch, start_step, min_loss

def save_state(epoch=0, step=0, performance=None, 
               model = None, optimizer=None, lr_scheduler=None, 
               loss_scaler=None, min_loss=np.inf, 
               path=None, only_model=False):
    unwrapper_model = model
    while hasattr(unwrapper_model,'module'):
        unwrapper_model = unwrapper_model.module
    model = unwrapper_model

    if only_model:
        states = {'model': model.state_dict(),'epoch': epoch, 'step':  step}
    else:
        states = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'epoch': epoch,
            'step':  step,
        }
    for name, val in performance.items():states[name] = val
    # in case the driver full, we first do save and then mv 
    tmp_path = str(path)+'.tmp'
    torch.save(states, tmp_path)
    os.system(f"mv {tmp_path} {path}")

def get_patch_location_index(center,img_shape,patch_range):
    #assert isinstance(patch_range,(tuple,list))
    patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range)
    # we want to get the patch index around center with the self.patch_range
    # For example, 
    #   (i-1,j-1) (i ,j-1) (i+1,j-1)
    #   (i-1,j ) (i ,j ) (i+1,j )
    #   (i-1,j+1) (i ,j+1) (i+1,j+1)
    # notice our data is on the sphere, this mean the center in H should be in [-boundary+patch_range, boundary-patch_range]
    # and the position in W is perodic.
    assert center[-2] >= (patch_range[-2]-1)//2
    assert center[-2] <= img_shape[-2] - (patch_range[-2]//2)
    delta  = [list(range(-(patch_range[0]//2),patch_range[0]//2 + patch_range[0]%2))] + \
             [list(range(-(patch_range[1]//2),patch_range[1]//2 + patch_range[1]%2))] 
    delta = np.meshgrid(*delta)
    pos  = [c+dc for c,dc in zip(center,delta)]
    pos[-1]= pos[-1]%img_shape[-1] # perodic
    pos = np.stack(pos).transpose(0,2,1)
    return pos

def get_center_around_indexes(patch_range,img_shape, h_range=None, w_range=None):
    #assert isinstance(patch_range,(tuple,list))
    patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range)
    hlist   = range((patch_range[-2]-1)//2, img_shape[-2] - patch_range[-2]//2 ) if h_range is None else h_range
    wlist   = range(img_shape[-1]) if w_range is None else w_range
    xes,yes = np.meshgrid(hlist,wlist)
    coor    = np.stack([xes,yes],-1).reshape(-1,2)
    indexes = np.array([np.stack(get_patch_location_index([x,y],img_shape,patch_range)) for x,y in coor] )
    indexes = indexes.reshape(len(wlist),len(hlist),2,*patch_range).transpose(1,0,2,3,4)
    coor    = coor.reshape(len(wlist),len(hlist),2).transpose(2,1,0)
    return coor, indexes

def get_patch_location_index_3D(center,img_shape,patch_range):
        #assert isinstance(patch_range,(tuple,list))
        patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range,patch_range)
        # we want to get the patch index around center with the patch_range
        # For example, 
        #   (i-1,j-1) (i ,j-1) (i+1,j-1)
        #   (i-1,j )  (i ,j )  (i+1,j )
        #   (i-1,j+1) (i ,j+1) (i+1,j+1)
        # notice our data is on the sphere, this mean the center in H should be in [-boundary+patch_range, boundary-patch_range]
        # and the position in W is perodic.
        assert center[-2] >= patch_range[-2]//2
        assert center[-2] <= img_shape[-2] - (patch_range[-2]//2)
        assert center[-3] >= patch_range[-3]//2
        assert center[-3] <= img_shape[-3] - (patch_range[-3]//2)
        delta  = [list(range(-(patch_range[0]//2),patch_range[0]//2+1))] + \
                 [list(range(-(patch_range[1]//2),patch_range[1]//2+1))] + \
                 [list(range(-(patch_range[2]//2),patch_range[2]//2+1))] 
        delta  = np.meshgrid(*delta)
        pos    = [c+dc for c,dc in zip(center,delta)]
        pos[-1]= pos[-1]%img_shape[-1] # perodic
        pos    = np.stack(pos).transpose(0,2,1,3) # !!!! <---- this is the important trick to align output patch range. The reason is np.meshgrid deal arguement wired.
        return pos

def get_center_around_indexes_3D(patch_range,img_shape,z_range=None, h_range=None, w_range=None):
    patch_range = patch_range if isinstance(patch_range,(list,tuple)) else (patch_range,patch_range,patch_range)
    assert isinstance(patch_range,(tuple,list))
    wlist   = range(img_shape[-1]) if w_range is None else w_range
    hlist   = range(patch_range[1]//2, img_shape[-2] - (patch_range[1]//2)) if h_range is None else h_range
    zlist   = range(patch_range[0]//2, img_shape[-3] - (patch_range[0]//2)) if z_range is None else z_range
    zes,yes,xes = np.meshgrid(zlist,hlist,wlist)
    coor    = np.stack([zes,yes,xes],-1).reshape(-1,3)
    indexes = np.array([np.stack(get_patch_location_index_3D([z,y,x],img_shape,patch_range)) for z,y,x in coor] )
    indexes = indexes.reshape(len(wlist),len(hlist),len(zlist),3,*patch_range).transpose(2,1,0,3,4,5,6)
    coor    = coor.reshape(len(wlist),len(hlist),len(zlist),3).transpose(3,2,1,0)
    return coor, indexes

from contextlib import contextmanager
@contextmanager
def optional_no_grad(flag):
    if flag:
        with torch.no_grad():
            yield
    else:
        yield

def get_sub_luna_point(time):
    import ephem
    greenwich = ephem.Observer()
    greenwich.lat = "0"
    greenwich.lon = "0"
    greenwich.date =time

    #add Moon Sub Solar Point
    moon = ephem.Moon(greenwich)
    moon.compute(greenwich.date)
    moon_lon = math.degrees(moon.ra - greenwich.sidereal_time() )
    # map longitude value from -180 to +180 
    if moon_lon < -180.0 :moon_lon = 360.0 + moon_lon 
    elif moon_lon > 180.0 :moon_lon = moon_lon - 360.0

    moon_lat = math.degrees(moon.dec)
    #print( "moon Lon:",moon_lon, "Lat:",moon_lat)
    return moon_lon,moon_lat

def get_sub_sun_point(time):
    import ephem
    greenwich      = ephem.Observer()
    greenwich.lat  = "0"
    greenwich.lon  = "0"
    greenwich.date  =time

    #add sun Sub Solar Point
    sun = ephem.Sun(greenwich)
    sun.compute(greenwich.date)
    sun_lon = math.degrees(sun.ra - greenwich.sidereal_time() )
    # map longitude value from -180 to +180 
    if sun_lon < -180.0 :sun_lon = 360.0 + sun_lon 
    elif sun_lon > 180.0 :sun_lon = sun_lon - 360.0

    sun_lat = math.degrees(sun.dec)
    #print( "sun Lon:",sun_lon, "Lat:",sun_lat)
    return sun_lon,sun_lat    

def get_local_rank(args=None):
    if args is None or args.Pengine.engine.name != 'naive_distributed':return int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    return args.Pengine.engine.local_rank



def get_tensor_norm(tensor, dim):  # <--use mse way
    #return (torch.sum(tensor**2,dim=dim)).sqrt()#(N,B)
    return (torch.mean(tensor**2, dim=dim))  # (N,B)

def dprint(*args, **kwargs):
    if get_local_rank():return
    print(*args, **kwargs)

def update_experiment_info(experiment_hub_path, epoch, args):
    try:
        if os.path.exists(experiment_hub_path):
            with open(experiment_hub_path, 'r') as f:
                experiment_hub = json.load(f)
        else:
            experiment_hub = {}
        path = str(args.SAVE_PATH)
        if path not in experiment_hub:
            experiment_hub[path] = {"id": len(
                experiment_hub), 'epoch_tot': args.Train.epochs, "start_time": time.strftime("%m_%d_%H_%M_%S")}
        experiment_hub[path]['epoch'] = epoch
        experiment_hub[path]['endtime'] = time.strftime("%m_%d_%H_%M_%S")
        with open(experiment_hub_path, 'w') as f:
            json.dump(experiment_hub, f)
    except:
        pass

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


    