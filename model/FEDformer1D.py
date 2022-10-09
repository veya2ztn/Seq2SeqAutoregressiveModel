import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import numpy as np
import torch
import torch.nn as nn


from .Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from .Embedding import DataEmbedding_wo_pos

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
    

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
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

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class FEDformer1D(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, img_size=None, in_chans=None, out_chans=None,embed_dim=None, depth=2,
               history_length=6, modes=3, mode_select='',label_len=3,pred_len=1,moving_avg=None,
               dropout=0,time_unit='h',n_heads=8,**kargs):
        super().__init__()
        self.mode_select            = mode_select
        self.modes                  = modes
        self.label_len              = label_len
        self.pred_len               = pred_len
        self.moving_avg             = [history_length//2] if moving_avg is None else moving_avg
        self.seq_len = seq_len      = history_length
        self.seq_len_dec = seq_len_dec = label_len+pred_len
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
        self.d_ff = None
        # Decomp
        kernel_size = self.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        Block_kargs={'in_channels':embed_dim,'out_channels':embed_dim,'modes':modes,'mode_select_method':mode_select}
    
        self.enc_embedding = DataEmbedding_wo_pos(self.in_chans, self.embed_dim, 'h',self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.in_chans, self.embed_dim, 'h',self.dropout)

        # if configs.version == 'Wavelets':
        #     encoder_self_att = MultiWaveletTransform(ich=self.embed_dim, L=configs.L, base=configs.base)
        #     decoder_self_att = MultiWaveletTransform(ich=self.embed_dim, L=configs.L, base=configs.base)
        #     decoder_cross_att = MultiWaveletCross(in_channels=self.embed_dim,
        #                                           out_channels=self.embed_dim,
        #                                           seq_len_q=self.seq_len // 2 + self.pred_len,
        #                                           seq_len_kv=self.seq_len,
        #                                           modes=self.modes,
        #                                           ich=self.embed_dim,
        #                                           base=configs.base,
        #                                           activation=configs.cross_activation)
        # else:
        encoder_self_att = FourierBlock(in_channels=self.embed_dim,
                                        out_channels=self.embed_dim,
                                        seq_len=self.seq_len,
                                        modes=self.modes,
                                        mode_select_method=self.mode_select)
        decoder_self_att = FourierBlock(in_channels=self.embed_dim,
                                        out_channels=self.embed_dim,
                                        seq_len=self.seq_len_dec,
                                        modes=self.modes,
                                        mode_select_method=self.mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=self.embed_dim,
                                                      out_channels=self.embed_dim,
                                                      seq_len_q=self.seq_len_dec,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len//2))
        dec_modes = int(min(self.modes, (self.seq_len//2+self.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att,self.embed_dim, self.n_heads),
                    self.embed_dim,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.depth)
            ],
            norm_layer=my_Layernorm(self.embed_dim)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att,self.embed_dim, self.n_heads),
                    AutoCorrelationLayer(decoder_cross_att,self.embed_dim, self.n_heads),
                    self.embed_dim,
                    self.out_chans,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.depth)
            ],
            norm_layer=my_Layernorm(self.embed_dim),
            projection=nn.Linear(self.embed_dim, self.out_chans, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
   
        channel_last = True
        if x_enc.shape[1:]==tuple([self.in_chans]+list(self.img_size)+[self.seq_len]):
            channel_last = False
            permute_order= [0]+list(range(2,len(x_enc.shape)))+[1]
            x_enc = x_enc.permute(*permute_order)

        ## x_enc      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T1, in_channels]
        ## x_mark_enc -->  [Batch,  T1, T_feature]
        ## x_dec      -->  [Batch,  *space_dims, in_channels] -> [Batch, z, h ,w, T2, in_channels]
        ## x_mark_dec -->  [Batch,  T2, T_feature]
        assert x_enc.shape[1:-1] == tuple(list(self.img_size)+[self.seq_len])
        assert x_mark_enc.shape[-2] == self.seq_len
        assert x_mark_dec.shape[-2] == self.seq_len_dec
        
        o_shape = x_enc.shape
        x_enc = x_enc.flatten(0,-3) ## --> x_enc -> [Batch*z*h*w, T1, in_channels]
        
        mean = torch.mean(x_enc, dim=1,keepdim=True)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :]]+[mean]*self.pred_len, dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        e_shape = dec_out.shape
        dec_out = dec_out.reshape(*o_shape[:-2],e_shape[-2],e_shape[-1])
        dec_out = dec_out[..., -self.pred_len:, :]
        if not channel_last:
            permute_order= [0,-1]+list(range(1,len(dec_out.shape)-1))
            dec_out = dec_out.permute(*permute_order)
        return   dec_out


