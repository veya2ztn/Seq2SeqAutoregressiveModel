import torch
import torch.nn as nn
from networks.utils.utils import PeriodicPad2d, ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish
# from networks.SwinUnet import SwinTransformer



class Encoder(nn.Module):
    def __init__(self, img_channel, latent_dim, ch=128, resolution=32, num_res_blocks=2, ch_mult=(1,1,2,2,4), attn_resolutions = [2]):
        super(Encoder, self).__init__()
        ch_mult = (1,) + tuple(ch_mult)
        layers = [
            PeriodicPad2d(1),
            nn.Conv2d(img_channel, ch * ch_mult[0], 3, 1, 0)]

        for i in range(len(ch_mult)-1):
            in_channels = ch * ch_mult[i]
            out_channels = ch * ch_mult[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                # if resolution in attn_resolutions:
                #     layers.append(NonLocalBlock(in_channels))
            if not (resolution in attn_resolutions):
            # if i != len(ch_mult) - 2:
                layers.append(DownSampleBlock(ch * ch_mult[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(ch * ch_mult[-1], ch * ch_mult[-1]))
        # layers.append(NonLocalBlock(ch * ch_mult[-1]))
        layers.append(ResidualBlock(ch * ch_mult[-1], ch * ch_mult[-1]))
        layers.append(GroupNorm(ch * ch_mult[-1]))
        layers.append(Swish())
        layers.append(PeriodicPad2d(1))
        layers.append(nn.Conv2d(ch * ch_mult[-1], latent_dim, 3, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class Encoder(nn.Module):
#     def __init__(self, img_channel, latent_dim, ch=128, resolution=32, num_res_blocks=2, ch_mult=(1,1,2,2,4), attn_resolutions = [2]):
#         super(Encoder, self).__init__()

#         self.layer_in = nn.Sequential(
#             PeriodicPad2d(1),
#             nn.Conv2d(img_channel, ch * ch_mult[0], 3, 1, 0),
#             ResidualBlock(ch * ch_mult[0], ch * ch_mult[1]),
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[1]),
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[1]),
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[1]),
#         )

#         self.down1 = DownSampleBlock(ch * ch_mult[1])
#         self.block1 = nn.Sequential(
#             ResidualBlock(ch * ch_mult[1], ch * ch_mult[2]),
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[2]),
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[2]),
#         )
#         self.bottom_out = nn.Conv2d(ch * ch_mult[2], latent_dim, 2, 2, 0)

        
#         self.down2 = DownSampleBlock(ch * ch_mult[2])
#         self.block2 = nn.Sequential(
#             ResidualBlock(ch * ch_mult[2], ch * ch_mult[3]),
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[3]),
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[3]),
#         )
    

#         self.block3 = nn.Sequential(
#             ResidualBlock(ch * ch_mult[3], ch * ch_mult[4]),
#             ResidualBlock(ch * ch_mult[4], ch * ch_mult[4]),
#             ResidualBlock(ch * ch_mult[4], ch * ch_mult[4]),
#             GroupNorm(ch * ch_mult[-1]),
#             Swish(),
#             PeriodicPad2d(1),
#             nn.Conv2d(ch * ch_mult[-1], latent_dim, 3, 1, 0)
#         )

#     def forward(self, x):
#         x = self.layer_in(x)
#         x = self.down1(x)
#         x = self.block1(x)
#         out1 = self.bottom_out(x.clone())
#         x = self.down2(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         res = torch.cat((out1, x), dim=-2)


#         return res




# class swin_Encoder(nn.Module):
#     def __init__(self, img_channel, latent_dim, ch=128, resolution=32, num_res_blocks=2, ch_mult=(1,1,2,2,4), attn_resolutions = [2]):
#         super(swin_Encoder, self).__init__()
#         self.model = SwinTransformer(1, in_chans=img_channel, embed_dim=64, depths=(2, 8, 8), num_heads=(4, 4, 8), window_size=(8, 8))

#     def forward(self, x):
#         return self.model(x)
