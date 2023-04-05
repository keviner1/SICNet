import torch
import torch.nn as nn
from einops import rearrange
# from .CrossUnet import CrossUnet
from .SICM import SICModule
from .Colorization import Colorization

class SICNet(nn.Module):
    def __init__(self, channels):
        super(SICNet, self).__init__()
        self.CrossUnet = SICModule(channels)
        self.CrossAttention = Colorization(channels)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  

    def forward(self, left, right):
        #add TR(right gt image) for directly warp
        TR = 0 
        # right = nn.functional.interpolate(right, scale_factor=4, mode='bicubic', align_corners=True)
        left_fea0, right_fea0, left_fea1, right_fea1, left_fea2, right_fea2, left_pre, right_pre = self.CrossUnet(left, right, TR)
        M, output = self.CrossAttention(left_fea0, right_fea0, left_fea1, right_fea1, left_fea2, right_fea2, left, left_pre, TR)
        return left_pre, right_pre, M, output
