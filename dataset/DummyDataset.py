
from .base import BaseDataset
import numpy as np

class SpeedTestDataset(BaseDataset):
    h = None
    w = None
    def __init__(self, split="train", config = {}):
        self.root         = config.get('root', self.default_root)
        self.years        = config.get('years', None)
        self.time_step    = config.get('time_step', 2)
        self.crop_coord   = config.get('crop_coord', None)
        self.channel_last = config.get('channel_last', None)
        self.with_idx     = config.get('with_idx', False)
        self.dataset_flag = config.get('dataset_flag', 'normal')
        self.use_time_stamp = config.get('use_time_stamp', False)
        self.time_reverse_flag = config.get('time_reverse_flag', False)
        self.channel_picks = config.get('channel_picks', list(range(70)))
        

    def __len__(self):
        return 1000

    def get_item(self, idx, reversed_part=False):
        arrays = np.random.randn(len(self.channel_picks), self.h, self.w)
        if reversed_part:
            arrays = arrays.copy() if isinstance(arrays, np.ndarray) else arrays.clone()  # it is a torch.tensor
            arrays[reversed_part] = -arrays[reversed_part]
        arrays = arrays[self.channel_pick]
        if self.crop_coord is not None:
            l, r, u, b = self.crop_coord
            arrays = arrays[:, u:b, l:r]
        #arrays = torch.from_numpy(arrays)
        if self.channel_last:
            arrays = arrays.transpose(1, 2, 0) if isinstance(
                arrays, np.ndarray) else arrays.permute(1, 2, 0)
        if self.use_time_stamp:
            return {'field': arrays, 'timestamp': self.timestamp[idx]}
        else:
            return arrays



class SpeedTestDataset32x64(SpeedTestDataset):
    h = 32
    w = 64


class SpeedTestDataset64x128(SpeedTestDataset):
    h = 64
    w = 128
