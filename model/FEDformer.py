import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .Embedding import *
from .afnonet import timer
#timer = #timer(True)
import copy

def get_symmetry_index_in_fft_pannel(shape):
    shape    = np.array(shape)
    indexes  = np.stack([t.flatten() for t in np.meshgrid(*[range(s) for s in shape])])
    indexes2 = (- indexes)%(shape[:,None])
    fftshift_index = (indexes + shape[:,None]//2)%shape[:,None]
    center   = (shape[:,None]-1)/2
    order    = np.argsort(np.linalg.norm(fftshift_index- center,axis=0))#sort by the distance to zero frequency
    indexes  = indexes[:,order]
    indexes2 =indexes2[:,order]
    indexes  = np.ravel_multi_index(indexes,(shape))
    indexes2 = np.ravel_multi_index(indexes2,(shape))
    return indexes,indexes2

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
        indexes,indexes2 = get_symmetry_index_in_fft_pannel(shape)
    
    rfft_freq = rfft_freq.flatten(*freq_dim)
    if mode == 'rfft':
        rfft_freq[...,indexes,:]= (rfft_freq[...,indexes,:] + rfft_freq[...,indexes2,:].conj())/2
    else:
        rfft_freq[...,indexes]= (rfft_freq[...,indexes] + rfft_freq[...,indexes2].conj())/2
    rfft_freq = rfft_freq.reshape(oshape)    
    return rfft_freq

def get_frequency_modes_mask_rfft(space_dims, modes=64, mode_select_method='',indexes=None,indexes2=None):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    #print(space_dims)
    rfft_freq_dim = list(space_dims)
    rfft_freq_dim[-1] = rfft_freq_dim[-1]//2 + 1
    if isinstance(modes,int):
        modes = [min(modes,dim) for dim in rfft_freq_dim]
    assert isinstance(modes,(list,tuple))
    space = np.array(space_dims[:-1])
    advise_picked_mode_num = 1 + np.sum(space//2) + np.ceil(np.prod(space-1)/2)
    totall_picked_mode_num = np.prod(modes[:-1])
    full_picked_mode_num = np.prod(space_dims)
    if mode_select_method == 'random':
        select_num = np.prod(rfft_freq_dim)
        needs_num  = np.prod(modes)
        turn_on_id = np.random.choice(range(select_num),needs_num,replace=False)
        mask_index = np.unravel_index(turn_on_id,rfft_freq_dim)
        mask = np.zeros(rfft_freq_dim)
        mask[mask_index]=1
        mask = torch.BoolTensor(mask)
    elif totall_picked_mode_num<full_picked_mode_num:
        totall_picked_mode_num  = min(totall_picked_mode_num,advise_picked_mode_num)
        if indexes is None:
            indexes,indexes2 = get_symmetry_index_in_fft_pannel(space)
        if totall_picked_mode_num < advise_picked_mode_num:
            print(f"for shape:{space} and pick modes:{modes[:-1]}<{advise_picked_mode_num}, we pick {totall_picked_mode_num} modes, the baseline modes is {advise_picked_mode_num}")
        else:
            print(f"for shape:{space} and pick modes:{modes[:-1]}>={advise_picked_mode_num}, we pick {totall_picked_mode_num} modes, the baseline modes is {advise_picked_mode_num}")
        picked_indexes=[]
        for i1,i2 in zip(indexes,indexes2):
            if i1 in picked_indexes:continue
            if i2 in picked_indexes:continue
            picked_indexes.append(i1)
            if len(picked_indexes)== totall_picked_mode_num:break

        mask = np.zeros((np.prod(rfft_freq_dim[:-1]),rfft_freq_dim[-1]))
        mask[picked_indexes,:modes[-1]]=1
        mask = mask.reshape(rfft_freq_dim)
        mask = torch.BoolTensor(mask)
    else:
        print(f"for shape:{space} and pick modes:{modes[:-1]}, we pick {totall_picked_mode_num} > total_modes {full_picked_mode_num}")
        print(f"in this case, we will use all modes!")
        mask = np.ones(rfft_freq_dim)
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
        #print(x.shape)
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

class SpaceTBatchNorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm3d(channels)

    def forward(self, x):
        assert len(x.shape)==5
        shape = x.shape
        BSpace_shape = shape[:-2]
        C = shape[-1]
        permute_order = [0,-1] + list(range(1,len(shape)-1))
        x = x.permute(*permute_order)#-->(B, *Space,T, C)-->(B, C, *Space,T)
        x = self.batchnorm(x)
        permute_order = [0] + list(range(2,len(shape))) + [1]
        x = x.permute(*permute_order)
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

class moving_avg_spacetime(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        assert len(kernel_size)==3
        self.kernel_size = np.array(kernel_size)
        self.avg         = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=0)
        self.pad_front   = self.kernel_size - 1-np.floor((self.kernel_size - 1) // 2)
        self.pad_end     = np.floor((self.kernel_size - 1) // 2)
        self.pad         = np.stack([self.pad_front,self.pad_end],1)[::-1].flatten().astype('int').tolist()
        #print(self.pad)
    
    def forward(self, x):
        # padding on the both ends of time series
        # the input must be (B,h,w,T,C)， the -1 dim is embed channel, the -2 dim is time channel
        assert len(x.shape)==5
        shape = x.shape
        BSpace_shape = shape[:-2]
        C = shape[-1]
        permute_order = [0,-1] + list(range(1,len(shape)-1))
        x = x.permute(*permute_order)#-->(B, *Space,T, C)-->(B, C, *Space,T)
        x = self.avg(F.pad(x,self.pad, mode='replicate'))
        permute_order = [0] + list(range(2,len(shape))) + [1]
        x = x.permute(*permute_order)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg_spacetime(kernel_size, stride=1)

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


########### FEDformer block ############

class FullFourierBlockN(nn.Module):
    """
    weight is (head, in_fea, out_fea, tokens) 
    thus is a very large tensor.
    """
    def __init__(self, in_channels=None, out_channels=None, space_dims=None, modes=None, 
               mode_select_method='random',
               head_num = 8,canonical_fft=True):
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

        indexes,indexes2 = get_symmetry_index_in_fft_pannel(shape)
        self.canonical_index1=indexes
        self.canonical_index2=indexes2

        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(head_num, in_channels // head_num, out_channels // head_num,
                                    self.mask.sum(), 
                                    dtype=torch.cfloat))
        self.canonical_fft=canonical_fft
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
        if self.canonical_fft:out_ft = canonical_fft_freq(out_ft,fft_dim,indexes=self.canonical_index1,indexes2=self.canonical_index2)
        x = torch.fft.irfftn(out_ft, dim=fft_dim, s=space_dims, norm='ortho')
        return (x, None)

class FourierBlockN(nn.Module):
    """
    weight is (head, in_fea, out_fea) 
    share transformation each token then make layer deep
    """
    def __init__(self, in_channels=None, out_channels=None, space_dims=None, modes=None, 
               mode_select_method='random',
               head_num = 8,canonical_fft=True,**kargs):
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

        indexes,indexes2 = get_symmetry_index_in_fft_pannel(shape)
        self.canonical_index1=indexes
        self.canonical_index2=indexes2

        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(torch.rand(head_num, in_channels // head_num, out_channels // head_num, 
                                    dtype=torch.cfloat)) #<----------------
        self.canonical_fft=canonical_fft
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
        out_ft[...,self.mask] =self.scale * torch.einsum('bhil,hio->bhol',x_ft[...,self.mask],self.weights1)
        # move scale here
        if self.canonical_fft:out_ft = canonical_fft_freq(out_ft,fft_dim,indexes=self.canonical_index1,indexes2=self.canonical_index2)
        x = torch.fft.irfftn(out_ft, dim=fft_dim, s=space_dims, norm='ortho')
        return (x, None)

########## Informer block ###########

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class LocalMask():
    def __init__(self, B, L,S,device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            self._mask = self._mask1+self._mask2
    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,**kargs):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # the input must be [Batch, head_num, in_channels, *space_dims]
        # the ouput must be [Batch, head_num, out_channels, *space_dims]
        queries_shape = queries.shape
        queries = queries.flatten(3,-1)
        B, H, D, L_Q = queries.shape
        queries = queries.permute(0,1,3,2)# B, H, L_Q, D

        key_shape = keys.shape
        keys    = keys.flatten(3,-1)
        _, _, _, L_K = queries.shape
        keys    = keys.permute(0,1,3,2)# B, H, L_K, D

        values_shape = values.shape
        values    = values.flatten(3,-1)
        _, _, _, L_V = queries.shape
        values    = values.permute(0,1,3,2)# B, H, L_V, D
 

        # queries = queries.transpose(2, 1) # B, H, L_Q, D
        # keys   = keys.transpose(2, 1)
        # values  = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        queries = queries/(np.sqrt(D)*np.sqrt(L_Q))
        keys  = keys/(np.sqrt(D)*np.sqrt(L_K))
        values = values/(np.sqrt(D)*np.sqrt(L_V))
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / (np.sqrt(D)*np.sqrt(L_Q))
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        context = context.permute(0,1,3,2).reshape(queries_shape)
        return context.contiguous(), attn

########### Common module ##############

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=1, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale 
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class CplxAdaptiveModReLU(nn.Module):
    r"""Applies soft thresholding to the complex modulus:
    $$
        F
        \colon \mathbb{C}^d \to \mathbb{C}^d
        \colon z \mapsto (\lvert z_j \rvert - \tau_j)_+
                        \tfrac{z_j}{\lvert z_j \rvert}
        \,, $$
    with $\tau_j \in \mathbb{R}$ being the $j$-th learnable threshold. Torch's
    broadcasting rules apply and the passed dimensions must conform with the
    upstream input. `CplxChanneledModReLU(1)` learns a common threshold for all
    features of the $d$-dim complex vector, and `CplxChanneledModReLU(d)` lets
    each dimension have its own threshold.
    """
    def __init__(self, *dim):
        super().__init__()
        self.dim = dim if dim else (1,)
        self.threshold = torch.nn.Parameter(torch.randn(*self.dim) * 0.02)

    def forward(self, input):
        modulus = torch.clamp(abs(input), min=1e-5)
        return input * torch.relu(1. - self.threshold / modulus)
        

    def __repr__(self):
        body = repr(self.dim)[1:-1] if len(self.dim) > 1 else repr(self.dim[0])
        return f"{self.__class__.__name__}({body})"


class FourierCrossAttentionN(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, space_dims_q=None, space_dims_kv=None, 
                 modes=64, mode_select_method='random', attention_stratagy='normal',
                 activation='tanh', policy=0,head_num=8,canonical_fft=True,**kargs):
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
        self.activation   = activation
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.space_dims_q =space_dims_q
        self.space_dims_kv=space_dims_kv
        self.attention_stratagy = attention_stratagy
        self.canonical_fft=canonical_fft
        # get modes for queries and keys (& values) on frequency domain
        
        self.mask_q  = get_frequency_modes_mask_rfft(space_dims_q, modes=modes, mode_select_method=mode_select_method)
        self.mask_kv = get_frequency_modes_mask_rfft(space_dims_kv, modes=modes, mode_select_method=mode_select_method)

        print(' modes_q={},  shape_q={}'.format( self.mask_q.sum(), self.mask_q.shape))
        print('modes_kv={}, shape_kv={}'.format(self.mask_kv.sum(),self.mask_kv.shape))

        shape    = np.array(space_dims_kv[:-1]) #omit the last one, which is half length from rfftn
        indexes,indexes2 = get_symmetry_index_in_fft_pannel(shape)
        self.canonical_index1=indexes
        self.canonical_index2=indexes2
        
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(head_num, in_channels // head_num, out_channels // head_num, self.mask_q.sum(), dtype=torch.cfloat))
        if self.activation == 'tanh':
            self.complex_nonlinear = nn.Tanh()
        elif self.activation == 'modReLU':
            self.complex_nonlinear = nn.Tanh()
    def normal_atttention(self,xq_ft_,xk_ft_):
        
        xqk_ft = (torch.einsum("...ex,...ey->...xy", xq_ft_, xk_ft_))
        #[Batch, head_num, in_channels, modes1] 
        #                       |                  --> [Batch, head_num, modes1, modes2] 
        #[Batch, head_num, in_channels, modes2] 
        #xqk_ft = self.complex_nonlinear(xqk_ft)
        xqk_ft = xqk_ft/abs(xqk_ft)
        # if self.activation == 'tanh':
        #     xqk_ft = xqk_ft.tanh()
        # elif self.activation == 'softmax':
        #     xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
        #     xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        # else:
        #     raise Exception('{} actiation function is not implemented'.format(self.activation))
                
        xqkv_ft = torch.einsum("...xy,...ey ->...ex", xqk_ft, xk_ft_)# <-- notice here is xk_ft rather than xv_ft
        ##[Batch, head_num,    modes1,   modes2]  
        #                                  |       --> [Batch, head_num, in_channels, modes1] 
        ##[Batch, head_num, in_channels, modes2] 
        return xqkv_ft

    def hydra_atttention(self,xq_ft_,xk_ft_):
        
        q = self.complex_nonlinear(xq_ft_)#--> [Batch, head_num,in_channels, modes1] 
        k = self.complex_nonlinear(xk_ft_)#--> [Batch, head_num,in_channels, modes1] 
        v = xk_ft_
        kv = torch.einsum("bhiy,bhjy ->bhij", k, v)
        ##[Batch, head_num, in_channels, modes2]  
        #                                  |       --> [Batch, head_num,in_channels, in_channels] 
        ##[Batch, head_num, in_channels, modes2] 
        qkv = (torch.einsum("bhjy,bhji->bhiy", q, kv))
        #[Batch, head_num, in_channels, modes1] 
        #                     |             --> [Batch, head_num*in_channels, modes1] 
        #[Batch, head_num, in_channels, in_channels] 
        qkv = qkv.reshape(xq_ft_.shape)
        return qkv

    def forward(self, q, k, v=None, mask=None):
        # the input must be [Batch, head_num, in_channels, *space_dims]
        space_dims_q= q.shape[3:]
        assert space_dims_q == self.space_dims_q
        space_dims_kv= k.shape[3:]
        assert space_dims_kv == self.space_dims_kv
        #timer.restart(2)
        fft_dim = tuple(range(3, 3 + len(space_dims_q)))
        xq_ft_ = torch.fft.rfftn(q, dim=fft_dim, norm='ortho')
        xq_ft_shape    = list(xq_ft_.shape)
        xq_ft_ = xq_ft_[...,self.mask_q]
        
        fft_dim = tuple(range(3, 3 + len(space_dims_kv)))
        xk_ft_ = torch.fft.rfftn(k, dim=fft_dim, norm='ortho')
        xk_ft_shape    = list(xk_ft_.shape)
        xk_ft_ = xk_ft_[...,self.mask_kv]
        #timer.record(f'fft','cross_attention',2)
        # xq_ft_ --> [Batch, head_num, in_channels, modes2] 
        # xk_ft_ --> [Batch, head_num, in_channels, modes2] 
        # usually the mode1/mode2 is very large, so that the attention between them is quite consuming
        
        # <(Batch, modes1, head_num, in_channels) | 
        #                               | 
        # (Batch, modes2, head_num, in_channels) >  
        #           |
        # (Batch, modes2, head_num, in_channels)


        if self.attention_stratagy == 'normal':
            xqkv_ft = self.normal_atttention(xq_ft_,xk_ft_)
        elif self.attention_stratagy == 'hydra':
            xqkv_ft = self.hydra_atttention(xq_ft_,xk_ft_)
        elif self.attention_stratagy == 'inflow':
            xqkv_ft = self.inflow_atttention(xq_ft_,xk_ft_)
        else:
            raise NotImplementedError("please assign correct attention stratagy")
        # the whole space dims are used to compute attention
        #timer.record(f'qkv','cross_attention',2)
        xqkvw   = torch.einsum("...ex,...eox->...ox", xqkv_ft, self.weights1)
        #timer.record(f'qkvw','cross_attention',2)
        # #[Batch, head_num, in_channels, modes1]   
        # #                       |                 --> [Batch, head_num, out_channels, modes1] 
        # #[Batch, head_num, in_channels, out_channel, modes1]         
        
        
        xq_ft_shape[2] = self.weights1.shape[2]
        out_ft         = torch.zeros(*xq_ft_shape, device=q.device, dtype=torch.cfloat)
        out_ft[...,self.mask_q] = xqkvw
        if self.canonical_fft:out_ft = canonical_fft_freq(out_ft,fft_dim,indexes=self.canonical_index1,indexes2=self.canonical_index2)
        #print(torch.abs(out_ft).min(),torch.abs(out_ft).max(),torch.std_mean(torch.abs(out_ft)))
        out    = torch.fft.irfftn(out_ft, dim=fft_dim, s=space_dims_q, norm='ortho')
        #timer.record(f'ifft','cross_attention',2)
        return (out, None)

class AutoCorrelationLayerN(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None,share_memory=True):
        super().__init__()

        self.d_model  = d_model
        self.d_keys   = d_keys   = d_keys or (d_model // n_heads)
        self.d_values = d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation if share_memory else copy.deepcopy(correlation)
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

        self.decomp = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', 
                                    bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x     [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        # cross [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        #timer.restart(3)
        space_dims = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        x = x + self.dropout(self.self_attention(x, x, x,attn_mask=x_mask)[0])
        x, trend1 = self.decomp(x)# --> [Batch, z, h ,w, T1, in_channels]
        #timer.record(f'self_attention','layers_decoder',3)
        x = x + self.dropout(self.cross_attention(x, cross, cross,attn_mask=cross_mask)[0])
        x, trend2 = self.decomp(x)# --> [Batch, z, h ,w, T1, in_channels]
        #timer.record(f'cross_attention','layers_decoder',3)
        x = x + self.dropout(self.distill_layer(x))  
        x, trend3 = self.decomp(x)
        #timer.record(f'decomp3','layers_decoder',3)
        residual_trend = trend1 + trend2 + trend3 #-->[Batch, z, h ,w, T1, in_channels]
        
        # do time
        residual_trend = residual_trend.transpose(-1, -2).flatten(0,-3)#-->[Batch*SpaceProd, in_channels, T1]
        residual_trend = self.projection(residual_trend).transpose(-1, -2).view(B,*space_dims,-1)
        #timer.record(f'projection','layers_decoder',3)
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

        self.decomp = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
  
    def forward(self, x, attn_mask=None):
        # x     [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        x, attn = self.attention(x, x, x,attn_mask=attn_mask)
        x       = x + self.dropout(x)
        x, _    = self.decomp(x)
        x       = x + self.distill_layer(x)
        x, _    = self.decomp(x)
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
        #timer.restart(1)
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
        #timer.record(f'layers_encoder','encoder',1)
        if self.norm is not None:
            x = self.norm(x) # 发散的元凶

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
        #timer.restart(1)
        for i,layer in enumerate(self.layers):
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        #timer.record(f'layers_decoder','decoder',1)
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
               history_length=6, modes=(17,33,6), mode_select='',label_len=3,pred_len=1,moving_avg=(5,5,3),
               dropout=0,time_unit='h',n_heads=8,canonical_fft=True,share_memory=True,**kargs):
        super(FEDformer, self).__init__()
        self.mode_select    = mode_select
        self.modes       = modes
        self.label_len     = label_len
        self.pred_len      = pred_len
        self.moving_avg     = (2,2,history_length//2) if moving_avg is None else moving_avg
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
        self.decomp = series_decomp(self.moving_avg)
        self.share_memory = share_memory
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        embedding_type = DataEmbedding_SpaceTimeCombine#DataEmbedding_SLSDTD
        self.enc_embedding = embedding_type(in_chans, embed_dim, len(self.space_dims_encoder) - 1, freq=time_unit, dropout=dropout)
        self.dec_embedding = embedding_type(in_chans, embed_dim, len(self.space_dims_encoder) - 1, freq=time_unit, dropout=dropout)
        
        Block_kargs={'in_channels':embed_dim,'out_channels':embed_dim,'modes':modes,
                     'mode_select_method':mode_select,'canonical_fft':canonical_fft,
                     'head_num':self.n_heads,
                     'factor':5, 'scale':None, 'attention_dropout':0.1, 'output_attention':False}
                     
        self.build_coder(Block_kargs)
    
    def build_coder(self,Block_kargs):

        encoder_self_att = FourierBlockN(space_dims=self.space_dims_encoder,**Block_kargs)
        decoder_self_att = FourierBlockN(space_dims=self.space_dims_decoder,**Block_kargs)
        decoder_cross_att= FourierCrossAttentionN(space_dims_q=self.space_dims_decoder,space_dims_kv=self.space_dims_encoder,**Block_kargs)
        embed_dim = Block_kargs['in_channels']
        self.encoder = Encoder(
            [
                EncoderLayerN(AutoCorrelationLayerN(encoder_self_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                              embed_dim, 
                              moving_avg=self.moving_avg,dropout=self.dropout,
                              activation=self.activation
                ) 
                for l in range(self.depth)
            ],
            norm_layer=SpaceTBatchNorm(embed_dim)#TLayernorm(embed_dim)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayerN(
                    AutoCorrelationLayerN(decoder_self_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                    AutoCorrelationLayerN(decoder_cross_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                    embed_dim,self.in_chans,
                    moving_avg=self.moving_avg,dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.depth)
            ],
            norm_layer=SpaceTBatchNorm(embed_dim),#TLayernorm(embed_dim),
            projection=nn.Linear(embed_dim, self.out_chans, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # >[decomp1]:cost 3.8e-04 ± 5.3e-04
        # >[enc_embedding]:cost 9.8e-04 ± 3.5e-04
        # >[encoder]:cost 8.5e-03 ± 5.2e-03
        # --[layers_encoder]:cost 8.3e-03 ± 5.1e-03
        # >[dec_embedding]:cost 1.2e-03 ± 1.1e-03
        # >[decoder]:cost 1.6e-02 ± 6.2e-03
        # --[layers_decoder]:cost 1.5e-02 ± 6.2e-03
        # ----[self_attention]:cost 3.3e-03 ± 2.5e-03
        # ----[cross_attention]:cost 3.9e-03 ± 4.2e-03
        # ------[fft_encoder]:cost 1.5e-03 ± 1.6e-03
        # ------[qkv]:cost 2.2e-04 ± 1.3e-04
        # ------[qkvw]:cost 1.2e-04 ± 3.6e-04
        # ------[ifft]:cost 1.3e-03 ± 1.7e-03
        # ----[decomp3]:cost 5.6e-04 ± 1.3e-03
        # ----[projection]:cost 3.8e-04 ± 4.5e-04
        if x_mark_dec.shape[1] == self.pred_len:
            # the input forget to cat the label len, then we do here
            x_mark_dec = torch.cat([x_mark_enc[:,-self.label_len:],x_mark_dec],1)

        # channel_last = 0
        # if x_enc.shape[1:]==tuple([self.in_chans]+list(self.img_size)+[self.seq_len]):
        #     channel_last = 1
        #     permute_order= [0]+list(range(2,len(x_enc.shape)))+[1]
        #     x_enc = x_enc.permute(*permute_order)
        # elif x_enc.shape[1:]==tuple([self.in_chans]+[self.seq_len]+list(self.img_size)):
        #     channel_last = 2
        #     permute_order= [0]+list(range(3,len(x_enc.shape)))+[2,1]
        #     x_enc = x_enc.permute(*permute_order)
        assert x_enc.shape[1:] == (*self.img_size, self.seq_len, self.in_chans)
        

        #timer.restart(0)
        ## x_enc      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        ## x_mark_enc -->  [Batch,  T1, 4]
        ## x_dec      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        ## x_mark_dec -->  [Batch,  T2, 4]
        
        seasonal_init, trend_init = self.decomp(x_enc)
        #timer.record('decomp1',level=0)
        # decoder input
        mean           = torch.mean(x_enc, dim=-2,keepdim=True)
        trend_init     = torch.cat([trend_init[..., -self.label_len:, :]]+[mean]*self.pred_len, dim=-2)
        seasonal_init  = F.pad(seasonal_init[..., -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        enc_out        = self.enc_embedding(x_enc, x_mark_enc)
        #timer.record('enc_embedding',level=0)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #timer.record('encoder',level=0)
        # dec
        
        dec_out                   = self.dec_embedding(seasonal_init, x_mark_dec)
        #timer.record('dec_embedding',level=0)
        
        seasonal_part, trend_part = self.decoder(dec_out, enc_out,x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)
        #timer.record('decoder',level=0)
        # # final

        dec_out = trend_part[...,:self.out_chans] + seasonal_part[...,:self.out_chans]

        # dec_out = dec_out[..., -self.pred_len:, :]
        dec_out = dec_out[..., -self.pred_len:, :]

        #dec_out  = F.pad(dec_out,(2,2))
        # [B, L, D]
        return dec_out

class Informer(FEDformer):
    """
    Informer
    """
    def build_coder(self,Block_kargs):
        encoder_self_att = ProbAttention(mask_flag=True, **Block_kargs)
        decoder_self_att = ProbAttention(mask_flag=True, **Block_kargs)
        decoder_cross_att= ProbAttention(mask_flag=False,**Block_kargs)
        embed_dim = Block_kargs['in_channels']
        self.encoder = Encoder(
            [
                EncoderLayerN(AutoCorrelationLayerN(encoder_self_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                        embed_dim, 
                        moving_avg=self.moving_avg,dropout=self.dropout,
                        activation=self.activation
                ) 
                for l in range(self.depth)
            ],
            norm_layer=SpaceTBatchNorm(embed_dim)#TLayernorm(embed_dim)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayerN(
                    AutoCorrelationLayerN(decoder_self_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                    AutoCorrelationLayerN(decoder_cross_att,embed_dim, self.n_heads,share_memory=self.share_memory),
                    embed_dim,self.in_chans,
                    moving_avg=self.moving_avg,dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.depth)
            ],
            norm_layer=SpaceTBatchNorm(embed_dim),#TLayernorm(embed_dim),
            projection=nn.Linear(embed_dim, self.out_chans, bias=True)
        )