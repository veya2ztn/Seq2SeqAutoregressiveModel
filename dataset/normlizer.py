import numpy as np
import torch

class DataNormlizer:
    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError
    
    def do_normlize_data(self, data):
        raise NotImplementedError
        

    def inv_normlize_data(self, data):
        raise NotImplementedError

    def do_normlize(self, data):
        if isinstance(data, list):
            return [self.do_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor,np.ndarray)):
            return self.do_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")
    
    def inv_normlize(self, data):
        if isinstance(data, list):
            return [self.inv_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return self.inv_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")

    @property
    def convert2samedtype(source_tensor, target_tensor):
        
        if isinstance(target_tensor, torch.Tensor):
            if isinstance(source_tensor, np.ndarray):
                return torch.from_numpy(source_tensor).to(target_tensor.device)
            elif isinstance(source_tensor, torch.Tensor) and target_tensor.device != source_tensor.device:
                return source_tensor.to(target_tensor.device)
        elif isinstance(target_tensor, np.ndarray) and isinstance(source_tensor, torch.Tensor):
            return source_tensor.detach().cpu().numpy()
        return source_tensor

    @property
    def aligned_for_batchtensor(source_tensor, target_tensor):
        if len(source_tensor.shape) + 1 == len(target_tensor.shape):
            return source_tensor[None]
        return source_tensor

    def align_tensor(self, source_tensor, target_tensor):
        source_tensor = self.convert2samedtype(source_tensor,target_tensor)
        source_tensor = self.aligned_for_batchtensor(source_tensor,target_tensor)
        return source_tensor
    
class GauessNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean, self.std = normlize_parameter
    
    @property
    def mean(self):
        return self.mean

    @property
    def std(self):
        return self.std
    
    def do_normlize_data(self, data):
        self.std = self.align_tensor(self.std ,data)
        self.mean= self.align_tensor(self.mean,data)
        return (data - self.mean)/self.std
        

    def inv_normlize_data(self, data):
        self.std = self.align_tensor(self.std,data)
        self.mean = self.align_tensor(self.mean, data)
        return data*self.std + self.mean
        
class UnitNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean, self.std = normlize_parameter
    
    @property
    def mean(self):
        return self.mean

    @property
    def std(self):
        return self.std
    
    def do_normlize_data(self, data):
        self.std = self.align_tensor(self.std,data)
        return data/self.std
        
    def inv_normlize_data(self, data):
        self.std = self.align_tensor(self.std, data)
        return data*self.std

class NoneNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean, self.std = normlize_parameter
    
    @property
    def mean(self):
        return self.mean

    @property
    def std(self):
        return self.std
    
    def do_normlize_data(self, data):
        return data
        
    def inv_normlize_data(self, data):
        return data
