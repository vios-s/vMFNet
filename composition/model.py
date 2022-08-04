import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationLayer(nn.Module):
    """Compute activation of a Tensor. The activation could be a exponent or a
    binary thresholding.
    """

    def __init__(self, vMF_kappa, compnet_type='vmf', threshold=0.0):
        super(ActivationLayer, self).__init__()
        self.vMF_kappa = vMF_kappa
        self.compnet_type = compnet_type
        self.threshold = threshold

    def forward(self, x):
        if self.compnet_type == 'vmf':
            x = torch.exp(self.vMF_kappa * x) * \
                (x > self.threshold).type_as(x)
        elif self.compnet_type == 'bernoulli':
            x = (x > self.threshold).type_as(x)
        return x

class Conv1o1Layer(nn.Module):
    def __init__(self, weights, device):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)
        self.device = device

    def forward(self, x):
        weight = self.weight
        xnorm = torch.norm(x, dim=1, keepdim=True)
        boo_zero = (xnorm == 0).type(torch.FloatTensor).to(self.device)
        xnorm = xnorm + boo_zero
        xn = x / xnorm
        wnorm = torch.norm(weight, dim=1, keepdim=True)
        weightnorm2 = weight / wnorm
        out = F.conv2d(xn, weightnorm2)
        if torch.sum(torch.isnan(out)) > 0:
            print('isnan conv1o1')
        return out



