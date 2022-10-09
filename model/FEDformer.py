import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .Embedding import *

def canonical_fft_freq(rfft_freq,freq_dim, mode='rfft', indexes=None,indexes2=None):
    '''
    Input: frequency coef as the result of torch.fftn or torch.rfftn
    
    ------------------------------------------------------------------
    This function aims to canonicalize the freqency mode before irfftn.
    The rfft can receive any complex number frequency value.
    It will omit the complex part to satisfy the Hermitian property.
    Literally, such a omiting operaion equation to a nonliear operation which is prohabit in some cases.
    -------------------------------------------------------------------
    The input is the torch.cfloat complex vector with the zero frequency at the beginning
    If the freqency vector is the result of rfftn
    Then the this method should not apply on the last dim.
    -------------------------------------------------------------------
    This symmetry is different in N-D fft processing.
    For example, in 3-D rfft, you can see the Hermitian property is cross reflection in [1:,1:] pannel
    a = torch.fft.rfftn(torch.randn(5,5,29))
    print(np.round(a[1:,1:,0].numpy(),2))
    [[-23.24+22.51j  11.31-34.77j  -2.47 -4.3j   22.49-17.21j]
     [-18.35-44.8j   12.29+13.86j  16.25+14.92j -25.59+30.88j]
     [-25.59-30.88j  16.25-14.92j  12.29-13.86j -18.35+44.8j ]
     [ 22.49+17.21j  -2.47 +4.3j   11.31+34.77j -23.24-22.51j]]
    '''
    if mode == 'rfft':
        freq_dim=freq_dim[:-1]
    oshape = rfft_freq.shape
    freq_dim=list(freq_dim)
    if indexes is None:
        shape    = np.array(oshape)[freq_dim]
        indexes  = np.stack([t.flatten() for t in np.meshgrid(*[range(s) for s in shape])])
        indexes2 = (- indexes)%(shape.reshape(2,1))
        indexes  = np.ravel_multi_index(indexes,(shape))
        indexes2 = np.ravel_multi_index(indexes2,(shape))
    
    rfft_freq = rfft_freq.flatten(*freq_dim)
    if mode == 'rfft':
        rfft_freq[...,indexes,:]= (rfft_freq[...,indexes,:] + rfft_freq[...,indexes2,:].conj())/2
    else:
        rfft_freq[...,indexes]= (rfft_freq[...,indexes] + rfft_freq[...,indexes2].conj())/2
    rfft_freq = rfft_freq.reshape(oshape)    
    return rfft_freq

def get_frequency_modes_mask_rfft(space_dims, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    
    rfft_freq_dim = list(space_dims)
    rfft_freq_dim[-1] = rfft_freq_dim[-1]//2 + 1
    if isinstance(modes,int):
        modes = [min(modes,dim) for dim in rfft_freq_dim]
    assert isinstance(modes,(list,tuple))
    if mode_select_method == 'random':
        select_num = np.prod(rfft_freq_dim)
        needs_num  = np.prod(modes)
        turn_on_id = np.random.choice(range(select_num),needs_num,replace=False)
        mask_index = np.unravel_index(turn_on_id,rfft_freq_dim)
        mask = np.zeros(rfft_freq_dim)
        mask[mask_index]=1
        mask = torch.BoolTensor(mask)
    else:
        mask       = np.zeros(rfft_freq_dim)
        if   len(modes)==1:mask[:modes[0]] = 1
        elif len(modes)==2:mask[:modes[0],:modes[1]] = 1
        elif len(modes)==3:mask[:modes[0],:modes[1],:modes[2]] = 1
        elif len(modes)==4:mask[:modes[0],:modes[1],:modes[2],:modes[3]] = 1
        else:raise NotImplementedError
        mask = torch.BoolTensor(mask)
    return mask

class TLayernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(TLayernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # x_hat = self.layernorm(x)
        # bias  = torch.mean(x_hat, dim=-2).unsqueeze(-2).repeat(1, x.shape[1], 1)
        shape = x.shape
        BSpace_shape = shape[:-2]
        C = shape[-1]
        x = x.flatten(0,-3)#-->(BSpace, T, C)
        x = self.layernorm(x)
        x = x - torch.mean(x, dim=1,keepdim=True)
        x = x.reshape(*BSpace_shape, -1, C )
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg     = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.pad_front  = self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2)
        self.pad_end   = math.floor((self.kernel_size - 1) // 2)
    def forward(self, x):
        # padding on the both ends of time series
        # the input must be (B,*Space,T,C)， the -1 dim is embed channel, the -2 dim is time channel
        # front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        # end  = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        # x = torch.cat([front, x, end], dim=1)
        shape = x.shape
        BSpace_shape = shape[:-2]
        C = shape[-1]
        x = x.flatten(0,-3).permute(0, 2, 1)#-->(BSpace, T, C)-->(BSpace, C, T)
        x = self.avg(F.pad(x,(self.pad_front,self.pad_end),mode='replicate'))
        x = x.permute(0, 2, 1).reshape(*BSpace_shape, -1, C )
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 

class series_decomp_along_time(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if isinstance(kernel_size, list):
            self.backbone = series_decomp_multi(kernel_size)
        else:
            self.backbone = series_decomp(kernel_size)
    
    def forward(self,x):
        return self.backbone(x)


class FourierBlockN(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, space_dims=None, modes=None, 
               mode_select_method='random',
               head_num = 8):
        super().__init__()
        print('fourier enhanced block used!')
        """
        ND Fourier block. It performs representation learning on frequency domain, 
        It does FFT, linear transform, and Inverse FFT.    
        The dims should be [Batch, head_num, in_channels,  *space_dims]
        For example, 
        - the 1D Fourier Block is [Batch, head_num, in_channels, length]
        - the 2D Fourier Block is [Batch, head_num, in_channels,  w,  h]
        """
        # get modes on frequency domain
        self.mask = get_frequency_modes_mask_rfft(space_dims, modes=modes, mode_select_method=mode_select_method)
        print(f"create a mode filter shape={self.mask.shape} with {self.mask.sum()} mode activate")
        # Notice in the original implement, authors project the picked mode to a dense frequency space 
        # For example, in 1D case, if we chose to use the mode [0,w1,0,0,w4,0,w6,....]
        # We will firstly gather them into [w1,w4,w6,....]
        #    then pad to the orginal size [w1,w4,w6,0,0,0,....]
        #    the do inverse Fourier Transform.
        #    so, it is not a mask implement that we mask some frequency mode 
        # Thus, the self.index is a tuple list like [(0,0,0),(0,0,1),...,] (in ND case)
        #    who represent the frequency mode we want to keep.
        # However, in N-D case, we cannot handle such frequency gathering processing.
        # the only way is putting each mode marked as (k_x,k_y,k_z) into its own position
        self.space_dims   = space_dims
        self.in_channels  = in_channels
        self.out_channels = out_channels
        shape    = np.array(space_dims[:-1]) #omit the last one, which is half length from rfftn
        indexes  = np.stack([t.flatten() for t in np.meshgrid(*[range(s) for s in shape])])
        indexes2 = (- indexes)%(shape.reshape(2,1))
        indexes  = np.ravel_multi_index(indexes,(shape))
        indexes2 = np.ravel_multi_index(indexes2,(shape))
        self.canonical_index1=indexes
        self.canonical_index2=indexes2

        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(head_num, in_channels // head_num, out_channels // head_num,
                                    self.mask.sum(), 
                                    dtype=torch.cfloat))
        
    def forward(self, x, k=None, v=None, mask=None):
        # the input must be [Batch, head_num, in_channels, *space_dims]
        
        space_dims = x.shape[3:]
        assert space_dims == self.space_dims
        fft_dim = tuple(range(3, 3 + len(space_dims)))
        x_ft = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        x_ft_shape = list(x_ft.shape)
        x_ft_shape[2] = self.weights1.shape[2]
        out_ft = torch.zeros(*x_ft_shape, device=x.device, dtype=torch.cfloat)
        # the x_ft[...,self.index] will convert 
        #[Batch, head_num, in_channels, *space_dims] --> [Batch, head_num, in_channels, L]
        out_ft[...,self.mask] = torch.einsum('bhil,hiol->bhol',x_ft[...,self.mask],self.weights1)
        out_ft = canonical_fft_freq(out_ft,fft_dim,indexes=self.canonical_index1,indexes2=self.canonical_index2)
        x = torch.fft.irfftn(out_ft, dim=fft_dim, s=space_dims, norm='ortho')
        return (x, None)

class FourierCrossAttentionN(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, space_dims_q=None, space_dims_kv=None, 
                 modes=64, mode_select_method='random',
                 activation='tanh', policy=0,head_num=8):
        super().__init__()
        print(' fourier enhanced cross attention used!')
        """
 
        ND Cross Attention layer. 
        It does FFT, linear transform, attention mechanism and Inverse FFT.    
        The input only need Q and K where the V=K.
        The dims should be [Batch, head_num, in_channels,  *space_dims]
        For example, 
        - the 1D Fourier Block is [Batch, head_num, in_channels, length]
        - the 2D Fourier Block is [Batch, head_num, in_channels,  w,  h]
    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dims_q =space_dims_q
        self.space_dims_kv=space_dims_kv
        # get modes for queries and keys (& values) on frequency domain
        self.mask_q  = get_frequency_modes_mask_rfft(space_dims_q, modes=modes, mode_select_method=mode_select_method)
        self.mask_kv = get_frequency_modes_mask_rfft(space_dims_kv, modes=modes, mode_select_method=mode_select_method)

        print(' modes_q={},  shape_q={}'.format( self.mask_q.sum(), self.mask_q.shape))
        print('modes_kv={}, shape_kv={}'.format(self.mask_kv.sum(),self.mask_kv.shape))

        shape    = np.array(space_dims_kv[:-1]) #omit the last one, which is half length from rfftn
        indexes  = np.stack([t.flatten() for t in np.meshgrid(*[range(s) for s in shape])])
        indexes2 = (- indexes)%(shape.reshape(2,1))
        indexes  = np.ravel_multi_index(indexes,(shape))
        indexes2 = np.ravel_multi_index(indexes2,(shape))
        self.canonical_index1=indexes
        self.canonical_index2=indexes2
        
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(head_num, in_channels // head_num, out_channels // head_num, self.mask_q.sum(), dtype=torch.cfloat))

    def forward(self, q, k, v=None, mask=None):
        # the input must be [Batch, head_num, in_channels, *space_dims]
        space_dims_q= q.shape[3:]
        assert space_dims_q == self.space_dims_q
        space_dims_kv= k.shape[3:]
        assert space_dims_kv == self.space_dims_kv
        
        fft_dim = tuple(range(3, 3 + len(space_dims_q)))
        xq_ft_ = torch.fft.rfftn(q, dim=fft_dim, norm='ortho')
        xq_ft_shape    = list(xq_ft_.shape)
        xq_ft_ = xq_ft_[...,self.mask_q]
        
        fft_dim = tuple(range(3, 3 + len(space_dims_kv)))
        xk_ft_ = torch.fft.rfftn(k, dim=fft_dim, norm='ortho')
        xk_ft_shape    = list(xk_ft_.shape)
        xk_ft_ = xk_ft_[...,self.mask_kv]
        
        # the whole space dims are used to compute attention
        # how every, follow the FNO spirit, attention in the space region is same as 
        # do conolution 
        xqk_ft = (torch.einsum("...ex,...ey->...xy", xq_ft_, xk_ft_))
        #[Batch, head_num, in_channels, modes1] 
        #                       |                  --> [Batch, head_num, modes1, modes2] 
        #[Batch, head_num, in_channels, modes2] 

        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))

        # attention should happen

        xqkv_ft = torch.einsum("...xy,...ey ->...ex", xqk_ft, xk_ft_)# <-- notice here is xk_ft rather than xv_ft
        #[Batch, head_num,    modes1,   modes2]  
        #                                  |       --> [Batch, head_num, in_channels, modes1] 
        #[Batch, head_num, in_channels, modes2] 
        xqkvw   = torch.einsum("...ex,...eox->...ox", xqkv_ft, self.weights1)
        #[Batch, head_num, in_channels, modes1]   
        #                       |                 --> [Batch, head_num, out_channels, modes1] 
        #[Batch, head_num, in_channels, out_channel, modes1]         
        
        xq_ft_shape[2] = self.weights1.shape[2]

        out_ft         = torch.zeros(*xq_ft_shape, device=q.device, dtype=torch.cfloat)
        out_ft[...,self.mask_q] = xqkvw
        out_ft = canonical_fft_freq(out_ft,fft_dim,indexes=self.canonical_index1,indexes2=self.canonical_index2)
        out    = torch.fft.irfftn(out_ft / self.in_channels / self.out_channels, dim=fft_dim, s=space_dims_q, norm='ortho')
        return (out, None)

class AutoCorrelationLayerN(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super().__init__()

        self.d_model  = d_model
        self.d_keys   = d_keys   = d_keys or (d_model // n_heads)
        self.d_values = d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection  = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection    = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection  = nn.Linear(d_model, d_values * n_heads)
        
        self.out_projection    = nn.Linear(d_values * n_heads, d_model)
        self.n_heads           = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # queries [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        # keys    [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        # values  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        H = self.n_heads
        BS_dims_L = queries.shape[:-1]
        BS_dims_S = keys.shape[:-1]
        tensor_rank = len(queries.shape)
        
        Space_last_order= [0,-2, -1] + list(range(1,tensor_rank-1)) 

        queries = self.query_projection(queries).view(*BS_dims_L, H, self.d_keys).permute(*Space_last_order)  #[Batch, head_num, in_channels, *space_dims]
        keys    =      self.key_projection(keys).view(*BS_dims_S, H, self.d_keys).permute(*Space_last_order)  #[Batch, head_num, in_channels, *space_dims]
        values  =  self.value_projection(values).view(*BS_dims_S, H, self.d_values).permute(*Space_last_order)#[Batch, head_num, in_channels, *space_dims]
        
        # for some module like Fourier Correlation Layer, the input only contain queries.
        # so we can omit some calculation
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )#[Batch, head_num, out_channels, *space_dims]-> [Batch,head_num, out_channels, z, h ,w, T]
        Feature_last_order=[0] + list(range(3,len(out.shape))) + [1,2]
        out = out.permute(Feature_last_order).reshape(*BS_dims_L, -1 ) 
        return self.out_projection(out), attn

class DecoderLayerN(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, 
                 moving_avg=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 2 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        self.distill_layer=nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff, bias=False),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model, bias=False)
        )

        self.decomp1 = series_decomp_along_time(moving_avg)
        self.decomp2 = series_decomp_along_time(moving_avg)
        self.decomp3 = series_decomp_along_time(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', 
                                    bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x     [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        # cross [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        space_dims = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        x = x + self.dropout(self.self_attention(x, x, x,attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)# --> [Batch, z, h ,w, T1, in_channels]
        
        x = x + self.dropout(self.cross_attention(x, cross, cross,attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)# --> [Batch, z, h ,w, T1, in_channels]
        
        x = x + self.dropout(self.distill_layer(x))  
        x, trend3 = self.decomp3(x)

        residual_trend = trend1 + trend2 + trend3 #-->[Batch, z, h ,w, T1, in_channels]
        
        # do time
        residual_trend = residual_trend.transpose(-1, -2).flatten(0,-3)#-->[Batch*SpaceProd, in_channels, T1]
        residual_trend = self.projection(residual_trend).transpose(-1, -2).view(B,*space_dims,-1)
        
        return x, residual_trend

class EncoderLayerN(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 4 * d_model
        self.attention = attention
        self.distill_layer=nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff, bias=False),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model, bias=False)
        )

        self.decomp1 = series_decomp_along_time(moving_avg)
        self.decomp2 = series_decomp_along_time(moving_avg)
        self.dropout = nn.Dropout(dropout)
  
    def forward(self, x, attn_mask=None):
        # x     [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        x, attn = self.attention(x, x, x,attn_mask=attn_mask)
        x       = x + self.dropout(x)
        x, _    = self.decomp1(x)
        x       = x + self.distill_layer(x)
        x, _    = self.decomp2(x)
        return x, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=0):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, img_size=None, in_chans=None, out_chans=None,embed_dim=None, depth=2,
               history_length=6, modes=(17,33,6), mode_select='',label_len=3,pred_len=1,moving_avg=None,
               dropout=0,time_unit='h',n_heads=8,**kargs):
        super(FEDformer, self).__init__()
        self.mode_select    = mode_select
        self.modes       = modes
        self.label_len     = label_len
        self.pred_len      = pred_len
        self.moving_avg     = [history_length//2] if moving_avg is None else moving_avg
        self.seq_len = seq_len = history_length
        seq_len_dec = label_len + pred_len
        self.space_dims_encoder     = tuple(list(img_size)+[seq_len])
        self.space_dims_decoder     = tuple(list(img_size)+[seq_len_dec])
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = 'tanh'
        self.in_chans = in_chans
        self.out_chans= out_chans
        self.img_size =img_size
        self.embed_dim=embed_dim
        self.depth = depth
        self.time_unit = time_unit
        # Decomp
        self.decomp = series_decomp_along_time(self.moving_avg)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_SLSDTD(in_chans, embed_dim, len(self.space_dims_encoder) - 1, freq=time_unit, dropout=dropout)
        self.dec_embedding = DataEmbedding_SLSDTD(in_chans, embed_dim, len(self.space_dims_encoder) - 1, freq=time_unit, dropout=dropout)
        
        Block_kargs={'in_channels':embed_dim,'out_channels':embed_dim,'modes':modes,'mode_select_method':mode_select}
        
        encoder_self_att = FourierBlockN(space_dims=self.space_dims_encoder,**Block_kargs)
        decoder_self_att = FourierBlockN(space_dims=self.space_dims_decoder,**Block_kargs)
        decoder_cross_att= FourierCrossAttentionN(space_dims_q=self.space_dims_decoder,space_dims_kv=self.space_dims_encoder,**Block_kargs)

        self.encoder = Encoder(
            [
                EncoderLayerN(AutoCorrelationLayerN(encoder_self_att,embed_dim, self.n_heads),
                              embed_dim, 
                              moving_avg=self.moving_avg,dropout=self.dropout,
                              activation=self.activation
                ) 
                for l in range(self.depth)
            ],
            norm_layer=TLayernorm(embed_dim)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayerN(
                    AutoCorrelationLayerN(decoder_self_att,embed_dim, self.n_heads),
                    AutoCorrelationLayerN(decoder_cross_att,embed_dim, self.n_heads),
                    embed_dim,self.out_chans,
                    moving_avg=self.moving_avg,dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.depth)
            ],
            norm_layer=TLayernorm(embed_dim),
            projection=nn.Linear(embed_dim, self.out_chans, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        channel_last = True
        if x_enc.shape[1:]==tuple([self.in_chans]+list(self.space_dims_encoder)):
            channel_last = False
            permute_order= [0]+list(range(2,len(x_enc.shape)))+[1]
            x_enc = x_enc.permute(*permute_order)
        ## x_enc      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        ## x_mark_enc -->  [Batch,  T1]
        ## x_dec      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        ## x_mark_dec -->  [Batch,  T2]
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        mean           = torch.mean(x_enc, dim=-2,keepdim=True)
        trend_init     = torch.cat([trend_init[..., -self.label_len:, :]]+[mean]*self.pred_len, dim=-2)
        seasonal_init  = F.pad(seasonal_init[..., -self.label_len:, :], (0, 0, 0, self.pred_len))
        enc_out        = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec

        dec_out                   = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out,x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[..., -self.pred_len:, :]
        # [B, L, D]
        if not channel_last:
            permute_order= [0,-1]+list(range(1,len(dec_out.shape)-1))
            dec_out = dec_out.permute(*permute_order)
        return dec_out