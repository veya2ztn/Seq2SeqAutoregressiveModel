import numpy as np
import os,io
from tqdm import tqdm


from petrel_client.client import Client
import petrel_client

client=Client(conf_path="~/petreloss.conf")
data_dir = 'weatherbench:s3://weatherbench/weatherbench32x64/npy'
year, hour = 1980,1
url = f"{data_dir}/{year}/{year}-{hour:04d}.npy"
with io.BytesIO(client.get(url)) as f:
    array_ceph = np.load(f)
# array_ceph = client.get(url)
# array_ceph = np.load(array_ceph).reshape(110,32,64)
print(array_ceph.shape)
# DATAROOT = "/nvme/zhangtianning/datasets/ERA5/numpy/"
# SAVEROOT = "/nvme/zhangtianning/datasets/ERA5/numpy_set_float32/"
# H5ROOT   = "/nvme/zhangtianning/datasets/ERA5/h5_set/"
# Years = {
#     'train': range(1979, 2016),
#     'valid': range(2016, 2018),
#     'test': range(2018, 2019),
#     'all': range(1979, 2022)

# }
# import h5py
# def save_h5(path,obj):
#     h5f = h5py.File(path, "w")
#     h5f.create_dataset("data", data=obj)
#     h5f.close()
# tag= 'train'

# # Nabla_cdot_V_l   = np.load(os.path.join(SAVEROOT,f"Nabla_cdot_V_l_{tag}.npy" ))
# year_data=Field  = np.load(os.path.join(SAVEROOT,f"all_data_{tag}.npy"       ))[...,1:-1,:]
# # Vphysics_dx_l    = np.load(os.path.join(SAVEROOT,f"Vphysics_dx_{tag}.npy"    ))
# # Vphysics_dy_l    = np.load(os.path.join(SAVEROOT,f"Vphysics_dy_{tag}.npy"    ))
# Field_dx       = np.load(os.path.join(SAVEROOT,f"Field_dx_{tag}.npy"       ))
# Field_dy       = np.load(os.path.join(SAVEROOT,f"Field_dy_{tag}.npy"       ))
# Dt= 6*3600
# u = Field[:,0:1]
# v = Field[:,1:2]
# T = Field[:,2:3]
# p = Field[:,3:4]

# Field_channel_mean    = np.array([2.7122362e+00,9.4288319e-02,2.6919699e+02,2.2904861e+04]).reshape(1,4,1,1,1)
# Field_channel_std     = np.array([9.5676870e+00,7.1177821e+00,2.0126169e+01,2.2861252e+04]).reshape(1,4,1,1,1)
# Field_Dt_channel_mean =    np.array([  -0.02293313,-0.04692488  ,0.02711264   ,7.51324121]).reshape(1,4,1,1,1)
# Field_Dt_channel_std  =  np.array([  8.82677214 , 8.78834556  ,3.96441518   ,526.15269219]).reshape(1,4,1,1,1)


# Field_dt    = Field[1:]-Field[:-1]
# pysics_part = (u*Field_dx + v*Field_dy)[:-1]*Dt 
# Field_Dt    = Field_dt + pysics_part
# print(Field_Dt.mean(axis=(0,2,3,4)))
# print(Field_Dt.std(axis=(0,2,3,4)))
# print("========================================")
# print(Field_Dt[:3].mean(axis=(0,2,3,4)))
# print(Field_Dt[:3].std(axis=(0,2,3,4)))
# print("========================================")
# Field_Dt    = (Field_Dt - Field_Dt_channel_mean)/Field_Dt_channel_std
# print(Field_Dt.mean(axis=(0,2,3,4)))
# print(Field_Dt.std(axis=(0,2,3,4)))
# print("========================================")
# print(Field_Dt[:3].mean(axis=(0,2,3,4)))
# print(Field_Dt[:3].std(axis=(0,2,3,4)))
# exit()
# # save_h5(os.path.join(H5ROOT,f"Nabla_cdot_V_l_{tag}.h5" ),Nabla_cdot_V_l )
# # save_h5(os.path.join(H5ROOT,f"all_data_{tag}.h5"       ),year_data_l   )
# # save_h5(os.path.join(H5ROOT,f"Vphysics_dx_{tag}.h5"    ),Vphysics_dx_l    )
# # save_h5(os.path.join(H5ROOT,f"Vphysics_dy_{tag}.h5"    ),Vphysics_dy_l    )
# # save_h5(os.path.join(H5ROOT,f"Field_dx_{tag}.h5"       ),Field_dx_l       )
# # save_h5(os.path.join(H5ROOT,f"Field_dy_{tag}.h5"       ),Field_dy_l       )


# # np.save(os.path.join(SAVEROOT2,f"Nabla_cdot_V_l_{tag}" ),Nabla_cdot_V_l      )
# # np.save(os.path.join(SAVEROOT2,f"all_data_{tag}"       ),year_data_l   )
# # np.save(os.path.join(SAVEROOT2,f"Vphysics_dx_{tag}"    ),Vphysics_dx_l    )
# # np.save(os.path.join(SAVEROOT2,f"Vphysics_dy_{tag}"    ),Vphysics_dy_l    )
# # np.save(os.path.join(SAVEROOT2,f"Field_dx_{tag}"       ),Field_dx_l       )
# # np.save(os.path.join(SAVEROOT2,f"Field_dy_{tag}"       ),Field_dy_l       )


# import json
# # mean_std_info = {
# #         "Nabla_cdot_V_l":{'mean':float(Nabla_cdot_V_l.mean()),'std':float(Nabla_cdot_V_l.std())},
# #         "year_data_l"   :{'mean':float(year_data_l.mean()   ),'std':float(year_data_l.std()   )},
# #         "Vphysics_dx_l" :{'mean':float(Vphysics_dx_l.mean() ),'std':float(Vphysics_dx_l.std() )},
# #         "Vphysics_dy_l" :{'mean':float(Vphysics_dy_l.mean() ),'std':float(Vphysics_dy_l.std() )},
# #         "Field_dx_l"    :{'mean':float(Field_dx_l.mean()    ),'std':float(Field_dx_l.std()    )},
# #         "Field_dy_l"    :{'mean':float(Field_dy_l.mean()    ),'std':float(Field_dy_l.std()    )},
# #         }
# # with open(os.path.join(SAVEROOT,'mean_std_info.json'),'w') as f:
# #     json.dump(mean_std_info,f)
# # with open(os.path.join(SAVEROOT,'mean_std_info.json'),'r') as f:
# #     mean_std_info = json.load(f)
# #
# #
# # Nabla_cdot_V_l = (Nabla_cdot_V_l - mean_std_info["Nabla_cdot_V_l"]['mean'])/mean_std_info["Nabla_cdot_V_l"]['std']
# # year_data_l    = (year_data_l    - mean_std_info["year_data_l"   ]['mean'])/mean_std_info["year_data_l"   ]['std']
# # Vphysics_dx_l  = (Vphysics_dx_l  - mean_std_info["Vphysics_dx_l" ]['mean'])/mean_std_info["Vphysics_dx_l" ]['std']
# # Vphysics_dy_l  = (Vphysics_dy_l  - mean_std_info["Vphysics_dy_l" ]['mean'])/mean_std_info["Vphysics_dy_l" ]['std']
# # Field_dx_l     = (Field_dx_l     - mean_std_info["Field_dx_l"    ]['mean'])/mean_std_info["Field_dx_l"    ]['std']
# # Field_dy_l     = (Field_dy_l     - mean_std_info["Field_dy_l"    ]['mean'])/mean_std_info["Field_dy_l"    ]['std']
# #
# # Nabla_cdot_V_l = Nabla_cdot_V_l.astype('float16')
# # year_data_l    = year_data_l.astype('float16')
# # Vphysics_dx_l  = Vphysics_dx_l.astype('float16')
# # Vphysics_dy_l  = Vphysics_dy_l.astype('float16')
# # Field_dx_l     = Field_dx_l.astype('float16')
# # Field_dy_l     = Field_dy_l.astype('float16')
# #
# # assert not np.isinf(Nabla_cdot_V_l).any()
# # assert not np.isinf(year_data_l   ).any()
# # assert not np.isinf(Vphysics_dx_l ).any()
# # assert not np.isinf(Vphysics_dy_l ).any()
# # assert not np.isinf(Field_dx_l    ).any()
# # assert not np.isinf(Field_dy_l    ).any()
# #
# # assert not np.isnan(Nabla_cdot_V_l).any()
# # assert not np.isnan(year_data_l   ).any()
# # assert not np.isnan(Vphysics_dx_l ).any()
# # assert not np.isnan(Vphysics_dy_l ).any()
# # assert not np.isnan(Field_dx_l    ).any()
# # assert not np.isnan(Field_dy_l    ).any()
# #
# # SAVEROOT2= "/nvme/zhangtianning/datasets/ERA5/numpy_set_float16/"
# # np.save(os.path.join(SAVEROOT2,f"Nabla_cdot_V_l_{tag}" ),Nabla_cdot_V_l      )
# # np.save(os.path.join(SAVEROOT2,f"all_data_{tag}"       ),year_data_l   )
# # np.save(os.path.join(SAVEROOT2,f"Vphysics_dx_{tag}"    ),Vphysics_dx_l    )
# # np.save(os.path.join(SAVEROOT2,f"Vphysics_dy_{tag}"    ),Vphysics_dy_l    )
# # np.save(os.path.join(SAVEROOT2,f"Field_dx_{tag}"       ),Field_dx_l       )
# # np.save(os.path.join(SAVEROOT2,f"Field_dy_{tag}"       ),Field_dy_l       )


# exit()

# assert not os.path.exists(os.path.join(SAVEROOT,f"all_data_{tag}"))

# Hdx = 6371000*np.sin(np.linspace(0,720,49)/720*np.pi)*2*np.pi/1440.0
# Hdx = Hdx.reshape(1,1,1,49,1)[...,1:-1,:]
# Hdy = 6371000*np.pi/720.0



# Vphysics_dx_l    =[]
# Vphysics_dy_l    =[]
# Field_dx_l       =[]
# Field_dy_l       =[]
# Field_dz_l       =[]
# year_data_l      =[]
# Nabla_cdot_V_l   =[]

# for year in tqdm(Years[tag]):
#     Nabla_cdot_V_l.append(np.load(os.path.join(DATAROOT,f"Nabla_cdot_V_l_{year}.npy" )))
#     year_data_l.append(np.load(os.path.join(DATAROOT,f"all_data_{year}.npy"       )))
#     Vphysics_dx_l.append(np.load(os.path.join(DATAROOT,f"Vphysics_dx_{year}.npy"    )))
#     Vphysics_dy_l.append(np.load(os.path.join(DATAROOT,f"Vphysics_dy_{year}.npy"    )))
#     Field_dx_l.append(np.load(os.path.join(DATAROOT,f"Field_dx_{year}.npy"       )))
#     Field_dy_l.append(np.load(os.path.join(DATAROOT,f"Field_dy_{year}.npy"       )))

# year_data_l       = np.concatenate(year_data_l      )
# Nabla_cdot_V_l    = np.concatenate(Nabla_cdot_V_l   )
# Vphysics_dx_l     = np.concatenate(Vphysics_dx_l    )
# Vphysics_dy_l     = np.concatenate(Vphysics_dy_l    )
# Field_dx_l        = np.concatenate(Field_dx_l       )
# Field_dy_l        = np.concatenate(Field_dy_l       )

# Vphysics_dx_l     = Vphysics_dx_l[...,1:-1,:]/Hdx
# Vphysics_dy_l     = Vphysics_dy_l[...,1:-1,:]/Hdy
# Field_dx_l        = Field_dx_l[...,1:-1,:]/Hdx
# Field_dy_l        = Field_dy_l[...,1:-1,:]/Hdy
# Nabla_cdot_V      = Field_dx_l[:,0:1] + Field_dy_l[:,1:2]

# year=tag
# np.save(os.path.join(SAVEROOT,f"Nabla_cdot_V_l_{year}" ),Nabla_cdot_V_l      )
# np.save(os.path.join(SAVEROOT,f"all_data_{year}"       ),year_data_l   )
# np.save(os.path.join(SAVEROOT,f"Vphysics_dx_{year}"    ),Vphysics_dx_l    )
# np.save(os.path.join(SAVEROOT,f"Vphysics_dy_{year}"    ),Vphysics_dy_l    )
# np.save(os.path.join(SAVEROOT,f"Field_dx_{year}"       ),Field_dx_l       )
# np.save(os.path.join(SAVEROOT,f"Field_dy_{year}"       ),Field_dy_l       )
