""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from models.unet_parts import *

###################################### maybe delete some skips
class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64) #64x288x288 x1
        self.down1 = Down(64, 128)  #128x144x144 x2
        self.down2 = Down(128, 256) #256x72x72 x3
        self.down3 = Down(256, 512) #512x36x36 x4
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) #512x18x18 x5
        self.up1 = Up(1024, 512 // factor, bilinear) #256x36x36 y1
        self.up2 = Up(512, 256 // factor, bilinear) #128x72x72 y2
        self.up3 = Up(256, 128 // factor, bilinear) #64x144x144 y3
        self.up4 = Up(128, 64, bilinear) #64x288x288 y4
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        logits = self.outc(y4)
        return [logits, x1, x2, x3, x4, x5, y1, y2, y3, y4]
        #Layer   # 0     1    2   3   4  5   6   7   8   9