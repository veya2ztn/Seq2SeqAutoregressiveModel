import torch.nn as nn
import functools
from networks.utils.utils import PeriodicPad2d





class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (https://arxiv.org/pdf/1611.07004.pdf)
    """

    def __init__(self, img_channels, num_filters_last=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 3
        padding_size = 1
        sequence = [PeriodicPad2d(padding_size),
                    nn.Conv2d(img_channels, num_filters_last, kernel_size, stride=2, padding=0),
                    nn.LeakyReLU(0.2)]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            sequence += [
                PeriodicPad2d(padding_size),
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size, 2 if i < n_layers else 1, padding=0),
                # nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size,
                        #   2 if i < n_layers else 1, padding_size, bias=use_bias),
                norm_layer(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
                PeriodicPad2d(padding_size),
                nn.Conv2d(num_filters_last * num_filters_mult, num_filters_last * num_filters_mult, kernel_size, 1, padding=0),
                # nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size,
                        #   2 if i < n_layers else 1, padding_size, bias=use_bias),
                norm_layer(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [PeriodicPad2d(padding_size)]
        sequence += [nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, stride=1, padding=padding_size)]
        # sequence += [nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, 1, padding_size)]
        self.model = nn.Sequential(*sequence)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



    def forward(self, x):
        return self.model(x)

