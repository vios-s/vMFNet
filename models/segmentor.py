import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *
from models.blocks import *
import torch


class Segmentor(nn.Module):
    def __init__(self, anatomy_out_channels, num_classes, layer=8, bilinear=True):
        super(Segmentor, self).__init__()

        self.anatomy_out_channels = 0
        self.num_classes = num_classes+1
        self.layer = layer

        if self.layer==6: # 36x36
            input_channels = 512 + self.anatomy_out_channels
            out_channels = 256
        elif self.layer == 7: # 72x72
            input_channels = 256 + self.anatomy_out_channels
            out_channels = 128
        elif self.layer==8: # 144x144
            input_channels =  12 + self.anatomy_out_channels
            out_channels = 64

        self.conv1 = DoubleConv(input_channels, out_channels)

        factor = 2 if bilinear else 1

        # self.up = Up(512, 256 // factor, bilinear)  # 128x72x72
        # self.up = Up(1024, 512 // factor, bilinear)  # 128x72x72
        self.up2 = Up(512, 256 // factor, bilinear) #128x72x72
        self.up3 = Up(256, 128 // factor, bilinear) #64x144x144
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 64)

        self.outc = OutConv(64, self.num_classes)

    def forward(self, content, features):
        if self.layer == 6:
            out = torch.cat([content, features[4].detach()], dim=1)
            out = self.conv(out)
            out = self.up2(content, features[3].detach())
            out = self.up3(out, features[2].detach())
            out = self.up4(out, features[1].detach())
        elif self.layer == 7:
            out = torch.cat([content, features[3].detach()], dim=1)
            out = self.conv(out)
            out = self.up3(content, features[2].detach())
            out = self.up4(out, features[1].detach())
        elif self.layer == 8:
            # out = torch.cat([content, features[2].detach()], dim=1)
            out = self.conv1(content)
            out = self.up4(out)
            out = self.conv2(out)
        else:
            out = self.conv1(content)

        out = self.outc(out)
        # out = F.softmax(out, dim=1)
        out = torch.sigmoid(out)
        return out