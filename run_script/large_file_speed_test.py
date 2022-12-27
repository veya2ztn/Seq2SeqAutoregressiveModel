import numpy as np
import time
now = time.time()
data = np.load("datasets/weatherbench/train_set.npy")
cost = time.time() - now
print(f"time cost:{cost}")
print(data.shape)
#print(data.mean(axis=(0,2,3)))