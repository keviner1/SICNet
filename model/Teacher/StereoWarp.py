import torch
import torch.nn as nn
from einops import rearrange
from .Unet import Unet
from .PAM import PAM

class StereoWarp(nn.Module):
    def __init__(self, channels):
        super(StereoWarp, self).__init__()
        self.Unet = Unet(channels)
        self.PAM = PAM(channels)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  

    def forward(self, left_gray, left_color, right_color, mode):
        left_fea = self.Unet(left_color)
        right_fea = self.Unet(right_color)
        M, left_new = self.PAM(left_fea, right_fea, left_gray, left_color, right_color, mode)
        return M, left_new

