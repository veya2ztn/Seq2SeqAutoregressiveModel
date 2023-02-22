import torch
import torch.nn as nn
from networks.utils.mlp import RCAB, RDCAB, MAGMlp
from networks.utils.utils import PeriodicPad2d
import torch.nn.functional as F



class CGB(nn.Module):
    def __init__(self, x_in_dim, y_in_dim, out_dim, window_size=[4, 8], 
                upsample_y=False, use_bias=True, drop=0.) -> None:
        super().__init__()

        if upsample_y:
            self.up_y = nn.ConvTranspose2d(y_in_dim, out_dim, 2, 2)
            y_in_dim = out_dim
        else:
            self.up_y = nn.Identity()

        self.fcx = nn.Linear(x_in_dim, out_dim)
        self.fcy = nn.Linear(y_in_dim, out_dim)

        self.cross_gate1 = MAGMlp(dim=out_dim, window_size=window_size, bias=use_bias, drop=drop, get_weight=True)
        self.cross_gate2 = MAGMlp(dim=out_dim, window_size=window_size, bias=use_bias, drop=drop, get_weight=True)
        self.fc1 = nn.Linear(out_dim, out_dim, bias=use_bias)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=use_bias)

        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, y):
        y = y.permute(0, 3, 1, 2)
        y = self.up_y(y)
        y = y.permute(0, 2, 3, 1)

        x = self.fcx(x)
        y = self.fcy(y)

        shortcut_x = x
        shortcut_y = y

        gx = self.cross_gate1(x)
        gy = self.cross_gate2(y)

        x = x * gy
        y = y * gx

        x = self.fc1(x)
        x = self.drop1(x)
        y = self.fc2(y)
        y = self.drop2(y)

        y = shortcut_y + y
        x = shortcut_x + x + y

        return x, y






class Base_Block(nn.Module):
    def __init__(self, in_dim, out_dim, window_size=[4, 8], reduction=4, use_bias=True, 
                    drop=0., skip=True, block_depth=2, use_endec=False, up_sample=False, 
                    down_sample=False, use_rdcab=False) -> None:
        super().__init__()

        self.use_endec = use_endec
        self.up_sample = up_sample
        self.down_sample = down_sample
        self.skip=skip

        if up_sample:
            self.Conv_up = nn.ConvTranspose2d(in_dim, out_dim, 2, 2)
            in_dim = out_dim
        else:
            self.Conv_up = nn.Identity()
        

        if skip:
            self.fc1 = nn.Linear(out_dim+in_dim, out_dim)
        else:
            self.fc1 = nn.Linear(in_dim, out_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(
                MAGMlp(out_dim, window_size=window_size, bias=use_bias, drop=drop)
            )
            if not use_rdcab:
                self.blocks.append(
                    RCAB(dim=out_dim, reduction=reduction)
                )
            else:
                self.blocks.append(
                    RDCAB(dim=out_dim, reduction=reduction)
            )
        

        if use_endec:
            self.CGB_block = CGB(out_dim, out_dim, out_dim, window_size=window_size, use_bias=use_bias, drop=drop)
        else:
            self.CGB_block = None
        
        if down_sample:
            self.pad = PeriodicPad2d(1)
            self.Conv_down = nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=2)
        

    def forward(self, x, skip=None, enc=None, dec=None):

        x = self.Conv_up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.skip:
            x = torch.cat([x, skip], dim=-1)
    
        x = self.fc1(x)
        shortcut_long = x
        for blk in self.blocks:
            x = blk(x)
        x = shortcut_long + x

        if enc is not None and dec is not None:
            x, _ = self.CGB_block(x, enc+dec)

        if self.down_sample:
            x_down = self.Conv_down(self.pad(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        else:
            x_down = None
        
        return x, x_down
        
        

class MAXIM(nn.Module):
    def __init__(self, in_chans=20, out_chans=20, embed_dim=32, window_size=[4,8], reduction=4, 
                use_bias=True, drop=0.1, enc_block_depth=2, dec_block_depth=2, bottle_block_depth=2, 
                num_stages=3, endeblock_num=3, bottleneck_num=2, num_supervision_scales=1) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_stages = num_stages
        self.endeblock_num = endeblock_num
        self.bottleneck_num = bottleneck_num
        self.num_supervision_scales = num_supervision_scales


        self.conv_blocks = nn.ModuleList()
        self.pad_blocks = nn.ModuleList()
        for i in range(num_supervision_scales):
            self.pad_blocks.append(PeriodicPad2d(1))
            self.conv_blocks.append(nn.Conv2d(in_chans, embed_dim*(2**i), 3, 1, 0))

        self.encoder_blocks = nn.ModuleList()
        for i in range(endeblock_num):
            encoder_block = Base_Block(
                in_dim=embed_dim*(2**(i-1)) if i > 1 else embed_dim,
                out_dim=embed_dim*(2**i),
                window_size=window_size,
                reduction=reduction,
                use_bias=use_bias,
                drop=drop,
                skip=True if i < self.num_supervision_scales else False,
                block_depth=enc_block_depth,
                use_endec=False if num_stages<=1 else True,
                up_sample=False,
                down_sample=True,
                use_rdcab=False
            )
            self.encoder_blocks.append(encoder_block)

        self.Bottlenecks = nn.ModuleList()
        for i in range(bottleneck_num):
            Bottleneck = Base_Block(
                in_dim=embed_dim*(2**(endeblock_num-1)) if i == 0 else embed_dim*(2**(endeblock_num)),
                out_dim=embed_dim*(2**(endeblock_num)),
                window_size=window_size,
                reduction=reduction,
                use_bias=use_bias,
                drop=drop,
                skip=False,
                block_depth=bottle_block_depth,
                use_endec=False,
                up_sample=False,
                down_sample=False,
                use_rdcab=True
            )
            self.Bottlenecks.append(Bottleneck)
        
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(endeblock_num)):
            decoder_block = Base_Block(
                in_dim=embed_dim*(2**(i+1)),
                out_dim=embed_dim*(2**i),
                window_size=window_size,
                reduction=reduction,
                use_bias=use_bias,
                drop=drop,
                skip=True,
                block_depth=dec_block_depth,
                use_endec=False,
                up_sample=True,
                down_sample=False,
                use_rdcab=False
            )
            self.decoder_blocks.append(decoder_block)

        self.to_cgb_ratio_lins = nn.ModuleList()
        for i in reversed(range(endeblock_num)):
            to_cgb_ratio_lin = nn.ModuleList()
            for j in range(endeblock_num):
                cgb_lin = nn.Linear(embed_dim*(2**j), embed_dim*(2**i))
                to_cgb_ratio_lin.append(cgb_lin)
            self.to_cgb_ratio_lins.append(to_cgb_ratio_lin)
        
        self.to_dec_ratio_lins = nn.ModuleList()
        self.to_dec_ratio_lins2 = nn.ModuleList()
        for i in reversed(range(endeblock_num)):
            to_dec_ratio_lin = nn.ModuleList()
            for j in range(endeblock_num):
                dec_lin = nn.Linear(embed_dim*(2**j), embed_dim*(2**i))
                to_dec_ratio_lin.append(dec_lin)
            self.to_dec_ratio_lins.append(to_dec_ratio_lin)
            self.to_dec_ratio_lins2.append(nn.Linear(embed_dim*(2**i)*endeblock_num, embed_dim*(2**i)))
        
        self.cgbs = nn.ModuleList()
        for i in reversed(range(endeblock_num)):
            cgb = CGB(
                    embed_dim*(2**i)*endeblock_num, 
                    embed_dim*(2**(i+1)), 
                    embed_dim*(2**i), 
                    window_size=window_size,
                    upsample_y=True,
                    use_bias=use_bias,
                    drop=drop)
            
            self.cgbs.append(cgb)

        self.final = nn.Linear(embed_dim, out_chans, bias=False)

    def forward(self, x):
        T = 1
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape
            x = x.unsqueeze(2)
        
        assert T==self.num_stages
        encs, decs = [], []
        for stage_idx in range(self.num_stages):
            decs_prev = []
            x_scales = []
            for j in range(self.num_supervision_scales):
                if j != 0:
                    x_scale = F.interpolate(x[:,:,stage_idx], (H//(2**j), W//(2**j)), mode='bilinear', align_corners=True)
                else:
                    x_scale = x[:, :, stage_idx]
                x_scale = self.pad_blocks[j](x_scale)
                x_scale = self.conv_blocks[j](x_scale).permute(0, 2, 3, 1) # B, H, W, C
                x_scales.append(x_scale)
            x_stage = x_scales[0]

            for j in range(self.endeblock_num):
                if stage_idx == 0:
                    bridge, x_stage = self.encoder_blocks[j](x_stage, x_scales[j])
                    encs.append(bridge)
                else:
                    bridge, x_stage = self.encoder_blocks[j](x_stage, x_scales[j], encs[j], decs[self.endeblock_num-j-1])
                    encs[j] = bridge
            
            for j in range(self.bottleneck_num):
                x_stage, _ = self.Bottlenecks[j](x_stage)
            
            global_feature = x_stage

            for j in range(self.endeblock_num):
                cgb_prev = []
                for i in range(self.endeblock_num):
                    cgb_prev.append(self.to_cgb_ratio_lins[j][i](F.interpolate(encs[i].permute(0,3,1,2), (H//(2**(self.endeblock_num-j-1)), W//(2**(self.endeblock_num-j-1))), mode='bilinear', align_corners=True).permute(0,2,3,1)))
                cgb_prev = torch.cat(cgb_prev, dim=-1)
                dec_prev, global_feature = self.cgbs[j](cgb_prev, global_feature)
                decs_prev.append(dec_prev)
            
            for j in range(self.endeblock_num):
                dec_bef = []
                for i in range(self.endeblock_num):
                    dec_bef.append(self.to_dec_ratio_lins[j][i](F.interpolate(decs_prev[self.endeblock_num-i-1].permute(0,3,1,2), (H//(2**(self.endeblock_num-j-1)), W//(2**(self.endeblock_num-j-1))), mode='bilinear', align_corners=True).permute(0,2,3,1)))
                dec_bef = torch.cat(dec_bef, dim=-1)
                dec_bef = self.to_dec_ratio_lins2[j](dec_bef)
                x_stage, _ = self.decoder_blocks[j](x_stage, dec_bef)
                if stage_idx == 0:
                    decs.append(x_stage)
                else:
                    decs[j] = x_stage
        res = self.final(x_stage).permute(0, 3, 1, 2)

        return res
            

            

                
                
        