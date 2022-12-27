import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

# import petrel_client
# from petrel_client.client import Client
# client_config_file = "/mnt/lustre/share/pymc/mc.conf"


# surface_pressure inf, mean_sea_level_pressure inf, 50h_geopotential inf
vnames = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure',
    '1000h_u_component_of_wind', '1000h_v_component_of_wind', '1000h_geopotential',
    '850h_temperature', '850h_u_component_of_wind', '850h_v_component_of_wind', '850h_geopotential', '850h_relative_humidity',
    '500h_temperature', '500h_u_component_of_wind', '500h_v_component_of_wind', '500h_geopotential', '500h_relative_humidity',
    '50h_geopotential',
    'total_column_water_vapour',
    # '850h_geopotential',
    # 'surface_pressure', 'mean_sea_level_pressure', '50h_geopotential',
    # '850h_geopotential'
]
vnames_short = [
    'u10', 'v10', 't2m', 'sp', 'msl',
    'u', 'v', 'z',
    't', 'u', 'v', 'z', 'r',
    't', 'u', 'v', 'z', 'r',
    'z',
    'tcwv'
    # 'z',
    # 'sp', 'msl', 'z',
    # 'z',
]
physics_index = [5, 9, 14, #u
                 6, 10, 15,#v
                 2, 8, 13, #T
                 7, 11, 16,#phi
                ]

years = range(2020, 2021)
#years = range(1988, 1989)
Ashape = (721, 1440)
import os
import subprocess

w = 721
h = 1440
#CACHEROOT = "/mnt/petrelfs/zhangtianning/dataset/ai4earth/EAR5Tiny20_720X1440/grid"
#DATAROOT  = "/mnt/petrelfs/zhangtianning/dataset/ai4earth/EAR5Tiny20_720X1440/numpy"
CACHEROOT = "/nvme/zhangtianning/datasets/ERA5/grid"
DATAROOT  = "/nvme/zhangtianning/datasets/ERA5/numpy"
def get_numpy_from_grid(vname_pair, year):
    if os.path.exists(os.path.join(DATAROOT,f"all_data_{year}.npy")):
        raise
    #return np.random.randn(1500,w,h)
    vname, vname_short = vname_pair
    fname   = f'{CACHEROOT}/{year}/{vname}/{vname:s}-{year:d}.grib'
    if len(os.listdir(CACHEROOT))>0:
        for tyear in os.listdir(CACHEROOT):
            if os.path.exists(os.path.join(DATAROOT,f"all_data_{tyear}.npy")):
                done_data_path = f"{CACHEROOT}/{tyear}"
                print(f"we have detected {tyear} data is finished, remove cache path:")
                print(f"{done_data_path}")
                os.system(f"rm -r {done_data_path}")
                #os.system(f"rm -r {CACHEROOT}/{year}")
                pass
    if not os.path.exists(fname):
        print(f"download {fname}")
        os.system(f"aws s3 --endpoint-url=http://10.140.2.254:80 --profile chenzhuo cp s3://era5/{vname}/{vname}-{year}.grib {fname}")
    ds = xr.open_dataset(fname, engine="cfgrib")
    ds_array = ds.data_vars[vname_short]
    return ds_array

def Drive(x,position=-1,dim=3,mode='circular'):
    expandQ=False
    if len(x.shape) == dim + 1 and x.shape[1]!=1:
        x = x.unsqueeze(1)
        expandQ = True
    conv_engine = [torch.conv1d, torch.conv2d, torch.conv3d][dim-1]
    weight = torch.Tensor([1/12, -8/12, 0, 8/12, -1/12]).cuda()
    weight = weight[(None,)*(dim + 1)].transpose(position,-1)
    pad_num = 2
    padtuple = [0]*dim*2
    padtuple[(-position-1)*2]   = pad_num
    padtuple[(-position-1)*2+1] = pad_num
    padtuple = tuple(padtuple)
    x = conv_engine(F.pad(x, padtuple, mode=mode), weight)
    return x.squeeze(1) if expandQ else x

def pad_data(year_data):
    nan_count= np.isnan(year_data).sum()
    nanlevel = 0
    _,w,h = year_data.shape
    if nan_count > 0:
        B, xx, yy = np.where(np.isnan(year_data))
        
        count =0 
        value = 0
        for left,right in [[xx-1,yy],
                           [xx+1,yy],
                           [xx,yy-1],
                           [xx,yy+1]
                          ]:
            left[left<0]+=w
            left[left>w-1]-=(w)
            right[right<0]+=h
            right[right>0]-=h
            value += year_data[B,left,right]
        year_data[B,xx,yy] = value/4
    
        
        while np.isnan(year_data).sum() > 0:
            print("bad nan detect")
            nanlevel=1 
            B, xx, yy = np.where(np.isnan(year_data))
            flag = False
            for step in range(1,10):
                for left,right in [[xx-step,yy],
                                   [xx+step,yy],
                                   [xx,yy-step],
                                   [xx,yy+step]
                                  ]:
                    left[left<0]+=w
                    left[left>w-1]-=(w)
                    right[right<0]+=h
                    right[right>0]-=h
                    if not np.isnan(year_data[B,left,right]).any():
                        year_data[B,xx,yy] = year_data[B,left,right]
                        flag = True
                        break
                if flag:break
            if flag:break
    return year_data
Dx = lambda x:Drive(x,position=-1,dim=3,mode='circular')
Dy = lambda x:Drive(x,position=-2,dim=3,mode='replicate')
Dz = lambda x:Drive(x,position=-3,dim=3,mode='replicate')
from tqdm import tqdm


x = np.linspace(0, 1440, 97)[:-1]
y = np.linspace(0, 720, 49)
x = x.astype('int')
y = y.astype('int')
x, y = np.meshgrid(x, y)
if __name__ == '__main__':
    for year in years:    
        print(f"loading {year} data ................", end="")
        year_datas= []
        for physics_name_idx in physics_index:
            vname = vnames[physics_name_idx]
            vname_short= vnames_short[physics_name_idx]
            year_datas.append(get_numpy_from_grid([vname, vname_short], year))
        #year_datas= np.stack(year_datas,1)
        print("done!")
        print(f"postprecessing ...............", end="")
        Vphysics_dx_l    =[]
        Vphysics_dy_l    =[]
        Field_dx_l       =[]
        Field_dy_l       =[]
        #Field_dz_l       =[]
        InteractionPart_l=[]
        year_data_l      =[]
        Nabla_cdot_V_l   =[]
        #year_datas= torch.Tensor(year_datas.reshape(len(year_datas),4,3,w,h))
        i=0
        nanlist= []
        for year_data in tqdm(zip(*year_datas)):
            i+=1
            year_data= np.stack(year_data,0)
            
            #break
            # nan_count= np.isnan(year_data).sum()
            # if nan_count > 0:
            #     p, xx, yy = np.where(np.isnan(year_data))
            #     year_data[p,xx,yy] = (year_data[p,xx-1,yy] + year_data[p,xx+1,yy] 
            #                         + year_data[p,xx,yy-1] + year_data[p,xx,yy+1])/4
            # nanlevel=0
            # if np.isnan(year_data).sum() > 0:
            #     print("bad nan detect")
            #     nanlevel=1 
            #     p, xx, yy = np.where(np.isnan(year_data))
            #     trail_x = xx
            #     while np.isnan(year_data[p,trail_x,yy]).any():
            #         trail_x+=1
            #     year_data[p,xx,yy] = year_data[p,trail_x,yy]
            # nanlist.append([np.isnan(year_data).sum(),nanlevel])
            year_data= pad_data(year_data)
            year_data= torch.Tensor(year_data).unsqueeze(0)
            
            year_data= year_data.reshape(1,4,3,w,h).cuda()
            u = year_data[:, 0:1]  # (Batch, 1, z, y ,x)
            v = year_data[:, 1:2]  # (Batch, 1, z, y ,x)
            T = year_data[:, 2:3]  # (Batch, 1, z, y ,x)
            p = year_data[:, 3:4]  # (Batch, 1, z, y ,x)
            V = torch.cat([u, v], 1)  # (Batch, 3, z, y ,x)
            Nabla_cdot_V    = (Dx(u[:, 0]) + Dy(v[:, 0])).unsqueeze(1)  # (Batch, 1, z, y ,x)
            Vphysics        = torch.stack([V*u, V*v, V*T, V*p], 1)
            Vphysics_dx     = Dx(Vphysics[:, :, 0].flatten(0, 1)).reshape(year_data.shape)  # (Batch, 4, z, y ,x)
            Vphysics_dy     = Dy(Vphysics[:, :, 1].flatten(0, 1)).reshape(year_data.shape)  # (Batch, 4, z, y ,x)
            Field_dx       = Dx(year_data.flatten(0,1)).reshape(year_data.shape)#(Batch, 4, z, y ,x)
            Field_dy       = Dy(year_data.flatten(0,1)).reshape(year_data.shape)#(Batch, 4, z, y ,x)
            #Field_dz       = Dz(year_data.flatten(0,1)).reshape(year_data.shape)#(Batch, 4, z, y ,x)
            InteractionPart = torch.stack([Dx(p[:, 0]),Dy(p[:, 0])], 1)  # (Batch,2, z, y ,x)
            
            Nabla_cdot_V_l.append(   (   Nabla_cdot_V[...,y,x].cpu().numpy()))
            year_data_l.append(      (      year_data[...,y,x].cpu().numpy()))
            Vphysics_dx_l.append(    (    Vphysics_dx[...,y,x].cpu().numpy()))
            Vphysics_dy_l.append(    (    Vphysics_dy[...,y,x].cpu().numpy()))
            Field_dx_l.append(       (       Field_dx[...,y,x].cpu().numpy()))
            Field_dy_l.append(       (       Field_dy[...,y,x].cpu().numpy()))
            #Field_dz_l.append(       (       Field_dz[...,y,x].cpu().numpy()))
            InteractionPart_l.append((InteractionPart[...,y,x].cpu().numpy()))        
    

        np.save(os.path.join(DATAROOT,f"Nabla_cdot_V_l_{year}" ),np.concatenate(Nabla_cdot_V_l   ))
        np.save(os.path.join(DATAROOT,f"all_data_{year}"       ),np.concatenate(year_data_l      ))
        np.save(os.path.join(DATAROOT,f"Vphysics_dx_{year}"    ),np.concatenate(Vphysics_dx_l    ))
        np.save(os.path.join(DATAROOT,f"Vphysics_dy_{year}"    ),np.concatenate(Vphysics_dy_l    ))
        np.save(os.path.join(DATAROOT,f"Field_dx_{year}"       ),np.concatenate(Field_dx_l       ))
        np.save(os.path.join(DATAROOT,f"Field_dy_{year}"       ),np.concatenate(Field_dy_l       ))
        #np.save(os.path.join(DATAROOT,f"Field_dz_{year}"       ),np.concatenate(Field_dz_l       ))
        np.save(os.path.join(DATAROOT,f"InteractionPart_{year}"),np.concatenate(InteractionPart_l))
