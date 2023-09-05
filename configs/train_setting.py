lr_for_mode={'pretrain':5e-4,'finetune':1e-4,'fourcast':1e-4}
ep_for_mode={'pretrain':80,'finetune':50,'fourcast':50}
bs_for_mode={'pretrain':4,'finetune':3,'fourcast':3}
as_for_mode={'pretrain':8,'finetune':16,'fourcast':16}
ts_for_mode={'pretrain':2,'finetune':3,'fourcast':36}

train_set={
    'large': ((720, 1440), 8, 20, 20, ERA5CephDataset,{}),
    'small': ( (32,   64), 8, 20, 20, ERA5CephSmallDataset,{}),
    'test_large': ((720, 1440), 8, 20, 20, lambda **kargs:SpeedTestDataset(720,1440,**kargs),{}),
    'test_small': ( (32,   64), 8, 20, 20, lambda **kargs:SpeedTestDataset( 32,  64,**kargs),{}),
    'physics_small': ( (32,   64), 2, 12, 12, ERA5CephSmallDataset,{'dataset_flag':'physics'}),
    'physics': ((720, 1440), 8, 12, 12, ERA5CephDataset,{'dataset_flag':'physics'}),
    '4796normal': ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normal','use_scalar_advection':False}),
    '4796Rad' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'reduce','use_scalar_advection':True}),
    '4796ad' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normal','use_scalar_advection':True}),
    '4796Field' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'only_Field','use_scalar_advection':False}),
    '4796FieldNormlized' : ((3,51,96), (1,3,3), 4, 4, ERA5Tiny12_47_96,{'dataset_flag':'normalized_data','use_scalar_advection':False}),
    '2D70N':((32,64), (2,2), 70, 70, WeathBench71,{'dataset_flag':'2D70N'}),
    '2D7066N':((32,64), (2,2), 70, 70, WeathBench7066,{'dataset_flag':'2D70N'}),
    '2D70NH5':((32,64), (2,2), 70, 70, WeathBench71_H5,{'dataset_flag':'2D70N'}),
    '2D706N':((32,64), (2,2), 70, 70, WeathBench706,{'dataset_flag':'2D70N'}), 
    '2D716N':((32,64), (2,2), 71, 71, WeathBench716,{'dataset_flag':'2D71N'}),
    '3D70N':((14,32,64), (2,2,2), 5, 5, WeathBench71,{'dataset_flag':'3D70N'}),
}