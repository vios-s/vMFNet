from .compcsd import *
from .unet_model import *
from .unet_parts import *
from .weight_init import *
from .blocks import *

import sys

def get_model(name, params):
    if name == 'compcsd':
        return CompCSD(params['image_channels'], params['layer'], params['vc_numbers'], params['num_classes'], params['z_length'],
                       params['anatomy_out_channels'], params['vMF_kappa'])
    if name == 'unet':
        return UNet(params['num_classes'])
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)