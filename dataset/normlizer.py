import numpy as np
import torch

class DataNormlizer:
    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError
    
    def do_pre_normlize_data(self, data):
        raise NotImplementedError("Implment a normlize way that inside the dataset object")
        
    def inv_pre_normlize_data(self, data):
        raise NotImplementedError("Implment a invnormlize way that inside the dataset object")

    def do_pre_normlize(self, data):
        if isinstance(data, list):
            return [self.do_pre_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor,np.ndarray)):
            return self.do_pre_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")
    
    def inv_pre_normlize(self, data):
        if isinstance(data, list):
            return [self.inv_pre_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return self.inv_pre_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")


    def do_post_normlize_data(self, data):
        raise NotImplementedError("Implment a normlize way that outside the dataset object")
        
    def inv_post_normlize_data(self, data):
        raise NotImplementedError("Implment a invnormlize way that outside the dataset object")


    def do_post_normlize(self, data):
        if isinstance(data, list):
            return [self.do_post_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor,np.ndarray)):
            return self.do_post_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")

    def inv_post_normlize(self, data):
        if isinstance(data, list):
            return [self.inv_post_normlize_data(d) for d in data]
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return self.inv_post_normlize_data(data)
        else:
            raise NotImplementedError(f"do not support data type {type(data)}")

    @staticmethod
    def convert2samedtype(source_tensor, target_tensor):
        
        if isinstance(target_tensor, torch.Tensor):
            if isinstance(source_tensor, np.ndarray):
                return torch.from_numpy(source_tensor).to(target_tensor.device)
            elif isinstance(source_tensor, torch.Tensor) and target_tensor.device != source_tensor.device:
                return source_tensor.to(target_tensor.device)
        elif isinstance(target_tensor, np.ndarray) and isinstance(source_tensor, torch.Tensor):
            return source_tensor.detach().cpu().numpy()
        return source_tensor

    @staticmethod
    def aligned_for_batchtensor(source_tensor, target_tensor):
        if len(source_tensor.shape) + 1 == len(target_tensor.shape):
            return source_tensor[None]
        return source_tensor

    def align_tensor(self, source_tensor, target_tensor):
        source_tensor = self.convert2samedtype(source_tensor,target_tensor)
        source_tensor = self.aligned_for_batchtensor(source_tensor,target_tensor)
        return source_tensor
    
class PreGauessNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean_tensor, self.std_tensor = normlize_parameter
    
    @property
    def mean(self):
        return self.mean_tensor

    @property
    def std(self):
        return self.std_tensor
    
    def do_pre_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor ,data)
        self.mean_tensor= self.align_tensor(self.mean_tensor,data)
        return (data - self.mean_tensor)/self.std_tensor
        
    def inv_pre_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor,data)
        self.mean_tensor = self.align_tensor(self.mean_tensor, data)
        return data*self.std_tensor + self.mean_tensor

    def do_post_normlize_data(self, data):
        return data
        
    def inv_post_normlize_data(self, data):
        return data

class PreUnitNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean_tensor, self.std_tensor = normlize_parameter
    
    @property
    def mean(self):
        return self.mean_tensor

    @property
    def std(self):
        return self.std_tensor
    
    def do_pre_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor,data)
        return data/self.std_tensor
        
    def inv_pre_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor, data)
        return data*self.std_tensor

    def do_post_normlize_data(self, data):
        return data
        
    def inv_post_normlize_data(self, data):
        return data
    
class NoneNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter=None):
        pass
    
    @property
    def mean(self):
        print("This is a NoneNormlizer, it has no mean")
        return None

    @property
    def std(self):
        print("This is a NoneNormlizer, it has no std")
        return None
    
    def do_pre_normlize_data(self, data):
        return data
        
    def inv_pre_normlize_data(self, data):
        return data

    def do_post_normlize_data(self, data):
        return data
        
    def inv_post_normlize_data(self, data):
        return data


class TimewiseNormlizer(NoneNormlizer):

    @property
    def mean(self):
        print("This is a TimewiseNormlizer, the mean depend on the real timestamp ")
        return None

    @property
    def std(self):
        print("This is a TimewiseNormlizer, the std  depend on the real timestamp")
        return None

    

class PostGauessNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean_tensor, self.std_tensor = normlize_parameter
    
    @property
    def mean(self):
        return self.mean_tensor

    @property
    def std(self):
        return self.std_tensor
    
    def do_post_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor ,data)
        self.mean_tensor= self.align_tensor(self.mean_tensor,data)
        return (data - self.mean_tensor)/self.std_tensor
        
    def inv_post_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor,data)
        self.mean_tensor = self.align_tensor(self.mean_tensor, data)
        return data*self.std_tensor + self.mean_tensor

    def do_pre_normlize_data(self, data):
        return data
        
    def inv_pre_normlize_data(self, data):
        return data

class PostUnitNormlizer(DataNormlizer):
    def __init__(self, normlize_parameter):
        self.mean_tensor, self.std_tensor = normlize_parameter
    
    @property
    def mean(self):
        return self.mean_tensor

    @property
    def std(self):
        return self.std_tensor
    
    def do_post_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor,data)
        return data/self.std_tensor
        
    def inv_post_normlize_data(self, data):
        self.std_tensor = self.align_tensor(self.std_tensor, data)
        return data*self.std_tensor

    def do_pre_normlize_data(self, data):
        return data
        
    def inv_pre_normlize_data(self, data):
        return data
 