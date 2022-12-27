from cephdataset import *
import h5py
def init_file_list(years):
    file_list = []
    for year in years:
        if year == 1979: # 1979年数据只有8753个，缺少第一天前7小时数据，所以这里我们从第二天开始算起
            for hour in range(17, 8753, 1):
                file_list.append([year, hour])
        else:
            if year % 4 == 0:
                max_item = 8784
            else:
                max_item = 8760
            for hour in range(0, max_item, 1):
                file_list.append([year, hour])
    return file_list
# split='train'
# dataset = WeathBench(split=split,dataset_flag='2D110N')

# np.save(f"{root}/{split}_set.npy",full_data)

root='datasets/weatherbench'
year=1979
for year in range(2018, 2019):
    yearhourlist= init_file_list([year])
    full_data = np.zeros((len(yearhourlist),110,32,64))
    print(f"{year}-->{full_data.shape}")
    for i,(year, hour) in tqdm(enumerate(yearhourlist)):
        url = f"{root}/{year}/{year}-{hour:04d}.npy"
        full_data[i]=np.load(url)

    with h5py.File(f"datasets/weatherbench_h5/{year}.hdf5","w") as f:
        d1=f.create_dataset("data",data=full_data)