import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from models.unet_model import *

class Encoder(nn.Module):
    def __init__(self, num_output_channels):
        super(Encoder, self).__init__()
        """
        Build an encoder to extract anatomical information from the image.
        """
        self.num_output_channels = num_output_channels

        self.unet = UNet(n_classes=self.num_output_channels)

    def forward(self, x):
        out = self.unet(x)
        return out