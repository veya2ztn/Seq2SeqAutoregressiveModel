import torch
import numpy as np 

#########################################
########### metric computing ############
#########################################
from torchvision.transforms import GaussianBlur


def generate_latweight(w, device):
    # steph = 180.0 / h
    # latitude = np.arange(-90, 90, steph).astype(np.int)
    tw = 32 if w < 32 else w
    latitude = torch.linspace(-np.pi/2, np.pi/2, tw).to(device)
    if w < 32:
        offset = (32 - w)//2
        latitude = latitude[offset:-offset]
    cos_lat = torch.cos(latitude)
    latweight = cos_lat/cos_lat.mean()
    latweight = latweight.reshape(1, w, 1, 1)
    return latweight


def compute_accu(ltmsv_pred, ltmsv_true):
    if len(ltmsv_pred.shape) == 5:ltmsv_pred = ltmsv_pred.flatten(1, 2)
    if len(ltmsv_true.shape) == 5:ltmsv_true = ltmsv_true.flatten(1, 2)
    ltmsv_pred = ltmsv_pred.permute(0, 2, 3, 1)
    ltmsv_true = ltmsv_true.permute(0, 2, 3, 1)
    # ltmsv_pred --> (B, w, h, property)
    # ltmsv_true --> (B, w, h, property)
    latweight = generate_latweight(ltmsv_pred.shape[1], ltmsv_pred.device)
    # history_record <-- (B, w,h, property)
    fenzi = (latweight*ltmsv_pred*ltmsv_true).sum(dim=(1, 2))
    fenmu = torch.sqrt((latweight*ltmsv_pred**2).sum(dim=(1, 2)) *
                       (latweight*ltmsv_true**2).sum(dim=(1, 2))
                       )
    return torch.clamp(fenzi/(fenmu+1e-10), 0, 10)


def compute_rmse(pred, true, return_map_also=False, smooth_sigma_1=0.1, smooth_sigma_2=2, smooth_times=0):

    if len(pred.shape) == 5:pred = pred.flatten(1, 2)
    if len(true.shape) == 5:true = true.flatten(1, 2)
    if smooth_times > 0:
        smother = GaussianBlur(3, sigma=(smooth_sigma_1, smooth_sigma_2))
    for i in range(smooth_times):
        pred = smother(pred)
    pred = pred.permute(0, 2, 3, 1)
    true = true.permute(0, 2, 3, 1)
    latweight = generate_latweight(pred.shape[1], pred.device)
    out = torch.sqrt(torch.clamp((latweight*(pred - true)**2).mean(dim=(1, 2)), 0, 1000))
    if return_map_also:
        out = [out, torch.clamp((latweight*(pred - true)**2).sum(dim=(0)), 0, 1000)]
    return out
