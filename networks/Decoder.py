import re
import torch
import torch.nn as nn
from networks.utils.utils import PeriodicPad2d, ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish
import math




class Decoder(nn.Module):
    def __init__(self, img_channel, latent_dim, ch=128, resolution=32, num_res_blocks=2, ch_mult=(1,1,2,2,4), attn_resolutions = [2]):
        super(Decoder, self).__init__()
        num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[num_resolutions-1]
        # curr_res = resolution // 2**(num_resolutions-1)
        curr_res = attn_resolutions[0]
        attn_num = num_resolutions - math.log(resolution//curr_res, 2)

        layers = [PeriodicPad2d(1),
                  nn.Conv2d(latent_dim, block_in, 3, 1, 0),
                  ResidualBlock(block_in, block_in),
                #   NonLocalBlock(block_in),
                  ResidualBlock(block_in, block_in)
                  ]

        for i in reversed(range(num_resolutions)):
            block_out = ch * ch_mult[i]
            for i_block in range(num_res_blocks + 1):
                layers.append(ResidualBlock(block_in, block_out))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     layers.append(NonLocalBlock(block_in))
            if i >= attn_num:
                layers.append(UpSampleBlock(block_in))
                curr_res = curr_res * 2

        layers.append(GroupNorm(block_in))
        layers.append(Swish())
        layers.append(PeriodicPad2d(1))
        layers.append(nn.Conv2d(block_in, img_channel, 3, 1, 0))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# class Decoder(nn.Module):
#     def __init__(self, img_channel, latent_dim, ch=128, resolution=32, num_res_blocks=2, ch_mult=(1,1,2,2,4), attn_resolutions = [2]):
#         super(Decoder, self).__init__()
        
#         ch_mult = list(reversed(ch_mult))

#         self.layer_in = nn.Sequential(
#             PeriodicPad2d(1),
#             nn.Conv2d(latent_dim, ch * ch_mult[0], 3, 1, 0),
#             ResidualBlock(ch * ch_mult[0], ch * ch_mult[0]),
#             ResidualBlock(ch * ch_mult[0], ch * ch_mult[0]),
#             ResidualBlock(ch * ch_mult[0], ch * ch_mult[1]),
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[1]),
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[1]),
#         )

#         self.up1 = UpSampleBlock(ch * ch_mult[1])
#         self.bottom_in = nn.ConvTranspose2d(latent_dim, ch * ch_mult[1], kernel_size=(2, 2), stride=(2, 2))

#         self.block1 = nn.Sequential(
#             ResidualBlock(ch * ch_mult[1]*2, ch * ch_mult[2]),
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[2]),
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[2]),
#         )

#         self.up2 = UpSampleBlock(ch * ch_mult[2])
#         self.block2 = nn.Sequential(
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[3]),
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[3]),
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[3]),
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[4]),
#             ResidualBlock(ch * ch_mult[4], ch * ch_mult[4]),
#             GroupNorm(ch * ch_mult[4]),
#             Swish(),
#             PeriodicPad2d(1),
#             nn.Conv2d(ch * ch_mult[4], img_channel, 3, 1, 0)
#         )

#     def forward(self, x):
#         bottom_x, top_x = torch.split(x, 8, dim=-2)
#         x = self.layer_in(top_x)
#         x = self.up1(x)
#         bottom_x = self.bottom_in(bottom_x)
#         x = torch.cat((x, bottom_x), dim=-3)
#         x = self.block1(x)
#         x = self.up2(x)
#         res = self.block2(x)

#         return res