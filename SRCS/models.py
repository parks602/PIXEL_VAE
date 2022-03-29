from torch import nn

from layers import *

import numpy as np
from test import minmaxNorm
class Gated(nn.Module):
    """
    Model combining several gated pixelCNN layers
    """

    def __init__(self, input_size, channels, num_layers, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                MaskedConv2d(
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 256*c, 1, groups=c)

    def forward(self, x):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xh, xv = x, x

        for layer in self.gated_layers:
            xv, xh = layer((xv, xh))

        x = self.conv2(xh)

        return x.view(b, c, 256, h, w).transpose(1, 2)

class LGated(nn.Module):
    """
    Gated model with location specific conditional
    """

    def __init__(self, input_size, conditional_channels, channels, num_layers, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                LMaskedConv2d(
                    (channels, h, w),
                    conditional_channels,
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 1, 1)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x1 = self.conv2(xh)
        x2 = self.conv2(xv)
        x = x1 + x2
        return x
        #return x.view(b, c, 256, h, w).transpose(1, 2)

class CGated(nn.Module):
    """
    Gated model with location-independent conditional
    """

    def __init__(self, input_size, cond_size, channels, num_layers, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                CMaskedConv2d(
                    (channels, h, w),
                    cond_size,
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 1, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x
        #return x.view(b, c, 256, h, w).transpose(1, 2)

class ImEncoder(nn.Module):
    """
    Encoder for a VAE
    """
    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=1, colors=1):
        a, b, c = 4, 8, 16  # channel sizes
        p, q, r = 1, 3, 7  # up/downsampling

        super().__init__()
        self.zsize = zsize

        # - Encoder
        modules = [
            util.Block(colors, a, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((p, p)),
            util.Block(a, b, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((q, q)),
            util.Block(b, c, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((r, r)),
        ]

        for i in range(depth):
            modules.append( util.Block(c, c, use_res=use_res, batch_norm=use_bn))

        modules.extend([
            util.Flatten(),
            nn.Linear((in_size[0] // (p*q*r)) * (in_size[1] //  (p*q*r)) * c, zsize * 2)
        ])

        self.encoder = nn.Sequential(*modules)

    def forward(self, image):
        zcomb = self.encoder(image)
        return zcomb[:, :self.zsize], zcomb[:, self.zsize:]

class ImDecoder(nn.Module):
    """
    Decoder for a VAE
    """
    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=1, out_channels=8):
        super().__init__()

        a, b, c = 4, 8, 16  # channel sizes
        p, q, r = 10, 18, 1  # up/downsampling

        self.zsize = zsize

        #- Decoder
        upmode = 'bilinear'
        modules = [
            nn.Linear(zsize, (in_size[0] // (p*q*r)) * (in_size[1] // (p*q*r)) * c), nn.LeakyReLU(),
            util.Reshape((c, in_size[0] // (p*q*r), in_size[1] // (p*q*r)))
        ]

        for _ in range(depth):
            modules.append( util.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn) )


        modules.extend([
            nn.Upsample(scale_factor=r, mode=upmode),
            util.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn),
            nn.Upsample(scale_factor=q, mode=upmode),
            util.Block(c, b, deconv=True, use_res=use_res, batch_norm=use_bn),
            nn.Upsample(scale_factor=p, mode=upmode),
            util.Block(b, a, deconv=True, use_res=use_res, batch_norm=use_bn),
            nn.ConvTranspose2d(a, out_channels, kernel_size=1, padding=0),
            #nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*modules)

    def forward(self, zsample):

        y_hat  = self.decoder(zsample)
        #y_hat  = minmaxNorm(y_hat, 0, 100)
        return(y_hat)


