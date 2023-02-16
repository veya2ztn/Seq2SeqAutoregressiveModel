import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from einops import rearrange
from .convNd import convNd
def FFT_for_Period(x, k=4):
    # [B, T, W, H ,C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(dim=(0,2,3,4))
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(dim=(2,3,4))[:, top_list]

def FFT_for_Period(x, k=3):
    # [B, T, W, H ,C]
    B = x.size(0)
    period = (2,3,4)
    xf   = torch.FloatTensor([1,1,1]).to(x.device)[None].repeat(B,1)
    return period, xf

class Inception_Block4D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block4D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_kernels  = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(convNd(in_channels, out_channels, kernel_size=2 * i + 1, padding=i,num_dims=4,is_transposed=False,use_bias=False))
        self.kernels = nn.ModuleList(kernels)
        # if init_weight:
        #     self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Conv2d,nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, seq_len,pred_len,top_k,d_model,d_ff,num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.k        = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block4D(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block4D(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, W, H ,C = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            real_len = self.seq_len + self.pred_len
            # padding
            if real_len % period != 0:
                length = ((real_len // period) + 1) * period
                padding = x[:, -(length - real_len):]
                out = torch.cat([x, padding], dim=1)# use the last padding
            else:
                length = real_len
                out = x
            # reshape
            out = rearrange(out.reshape(B, length // period, period, W, H ,C),'B L P W H C-> B C L P W H').contiguous()
            # 4D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = rearrange(out,'B C L P W H -> B (L P) W H C')
            res.append(out[:, :real_len])
        res = torch.stack(res, dim=1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.reshape(B,-1,1,1,1,1)
        res = torch.sum(res * period_weight, 1)
        # residual connection
        res = res + x
        return res

class TimesNet4D(nn.Module):
    def __init__(self, in_chans=70,out_chans=70, history_length=5,label_len=0,pred_len=5,model_depth=3,top_k=3,embed_dim=128,num_kernels=1,**kargs):
        super(TimesNet4D, self).__init__()
        
        self.seq_len  =seq_len = history_length
        self.label_len     = label_len
        self.pred_len      = pred_len
        self.layer         = e_layers = model_depth
        self.model         = nn.ModuleList([TimesBlock(seq_len,pred_len,top_k,embed_dim,embed_dim,num_kernels) for _ in range(e_layers)])

        
        self.layer_norm    = nn.LayerNorm(embed_dim)
        
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

        self.enc_embedding  = nn.Linear(in_chans, embed_dim)
        self.projection     = nn.Linear(embed_dim, out_chans, bias=True)

    def forward(self, x):
        ############Normalization from Non-stationary Transformer
        means,std = torch.std_mean(x, dim=2, keepdim=True) #(B 1 P W H)
        means,std = means.detach(),std.detach()+1e-2
        x     = (x - means)/std
        x     = rearrange(x,'B P T W H -> B T W H P')
        x     = self.enc_embedding(x) # (B,L,32,64,128)
        x     = rearrange(x,'B T W H P -> B P W H T')
        x     = self.predict_linear(x)# (B,128,32,64,L)
        x     = rearrange(x,'B P W H T -> B T W H P')# (B,L,32,64,128)
        # TimesNet
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))# (B,L,32,64,128)
        x = self.projection(x)# (B,L,32,64,70)
        x = rearrange(x,'B T W H P -> B P T W H')# (B,L,128,32,64)
        #### De-Normalization from Non-stationary Transformer
        x = x * std + means
        return x[:, :, -self.pred_len:]


