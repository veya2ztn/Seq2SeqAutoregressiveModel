import numpy as np 
import os
def read_npy_from_ceph(client, url, Ashape=(720, 1440)):
    try:
        array_ceph = client.get(url)
        array_ceph = np.frombuffer(array_ceph, dtype=np.half).reshape(Ashape)
    except:
        os.system(f"echo '{url}' >> fail_data_path.txt")
        raise NotImplementedError(f"fail to read ceph path {url}")
    return array_ceph


def read_npy_from_buffer(path, Ashape=(720, 1440)):
    buf = bytearray(os.path.getsize(path))
    with open(path, 'rb') as f:f.readinto(buf)
    return np.frombuffer(buf, dtype=np.half).reshape(Ashape)


def load_numpy_from_url(client, url):
    if "s3://" in url:
        array = read_npy_from_ceph(client, url)
    elif os.path.isfile(url):
        try:
            array = np.load(url)
        except ValueError:
            array = read_npy_from_buffer(url)
    else:
        raise ValueError(f"Invalid URL or path: {url}")
    return array