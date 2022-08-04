import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Decoder(nn.Module):
    def __init__(self, num_channels, anatomy_out_channels, layer=8):
        super(Decoder, self).__init__()

        self.num_channels = num_channels
        self.anatomy_out_channels = 0
        self.layer = layer

        self.up2 = deconv_bn_relu(256, 128, kernel_size=4, stride=2, padding=1)
        self.double_conv1 = DoubleConv(128 + self.anatomy_out_channels, 128)
        self.up3 = nn.ConvTranspose2d(128+self.anatomy_out_channels, 128+self.anatomy_out_channels, kernel_size=2, stride=2)
        self.double_conv2 = DoubleConv(128+self.anatomy_out_channels, 64)
        # self.up3 = deconv_bn_relu(128, 64, kernel_size=4, stride=2, padding=1)
        # self.up4 = deconv_bn_relu(64, 64, kernel_size=4, stride=2, padding=1)
        self.double_conv3 = DoubleConv(64 + self.anatomy_out_channels, 64)
        self.up4 = nn.ConvTranspose2d(64+self.anatomy_out_channels, 64+self.anatomy_out_channels, kernel_size=2, stride=2)
        self.double_conv4 = DoubleConv(64+self.anatomy_out_channels, 64)
        self.outc = nn.Conv2d(64, self.num_channels, kernel_size=1)

    def forward(self, features):
        if self.layer == 6:
            out = self.up2(features)
            out = self.up3(out)
            out = self.up4(out)
        elif self.layer == 7:
            out = self.double_conv1(features)
            out = self.up3(out)
            out = self.double_conv2(out)
            out = self.up4(out)
            out = self.double_conv4(out)
        elif self.layer == 8:
            out = self.double_conv3(features)
            out = self.up4(out)
            out = self.double_conv4(out)
        else:
            raise (RuntimeError("No defined layers in segmentor!"))

        out = self.outc(out)
        return out