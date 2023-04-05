import torch
import torch.nn as nn
from einops import rearrange
from .ResBlock import ResBlock,HinBlock
import cv2
import numpy as np

class SICModule(nn.Module):
    def __init__(self, channels):
        super(SICModule, self).__init__()
        #==============================input================================    
        self.conv_L = nn.Sequential(nn.Conv2d(1, channels, 3, 1, 1), nn.GroupNorm(16,channels), nn.ReLU(inplace=True))
        self.conv_R = nn.Sequential(nn.Conv2d(3, channels, 3, 1, 1), nn.GroupNorm(16,channels), nn.ReLU(inplace=True))
        #==============================encode==============================
        self.layer_dowm1 = Basicblock(channels,"down",2)
        self.layer_dowm2 = Basicblock(channels,"down",4)
        self.layer_dowm3 = Basicblock(channels,"down",8)
        self.layer0 = Basicblock(channels,"no",0)
        #==============================decode==============================
        self.layer_up3 = Basicblock(channels,"up",8)
        self.layer_up2 = Basicblock(channels,"up",4)
        self.layer_up1 = Basicblock(channels,"up",2)
        #==============================skip================================    
        self.fus1 = SkipFusion(channels)
        self.fus2 = SkipFusion(channels)
        self.fus3 = SkipFusion(channels)
        self.skip1 = SkipConnection(channels)
        self.skip2 = SkipConnection(channels)
        self.skip3 = SkipConnection(channels)
        #==============================SFM SCM block========================    
        self.color_transfer1 = SCM_block(channels)
        self.frequency_transfer1 = SFM_block(channels)
        #===================pre-reconstruction constranint==================
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1,bias=False),
                                      nn.GroupNorm(16, channels),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, 3, 1, 1, 0, bias=False))

    def forward(self, left, right, pe):
        left_encode3 = self.conv_L(left)
        right_encode3 = self.conv_R(right)

        #==============================encode==============================
        left_encode2 = self.layer_dowm1(left_encode3)
        right_encode2 = self.layer_dowm1(right_encode3)
        #-----------
        left_encode1 = self.layer_dowm2(left_encode2)
        right_encode1 = self.layer_dowm2(right_encode2)
        #-----------
        left_encode0 = self.layer_dowm3(left_encode1)
        right_encode0 = self.layer_dowm3(right_encode1)
        #==============================decode==============================
        left_decode0 = self.layer0(left_encode0)
        right_decode0 = self.layer0(right_encode0)
        left_decode1 = self.layer_up3(left_decode0)
        right_decode1 = self.layer_up3(right_decode0)
        left_decode1 = self.skip1(left_encode0, left_decode1)
        right_decode1 = self.skip1(right_encode0, right_decode1)
        #-----------
        left_decode1 = self.fus1(left_decode1,left_encode1)
        right_decode1 = self.fus1(right_decode1,right_encode1)
        left_decode2 = self.layer_up2(left_decode1)
        right_decode2 = self.layer_up2(right_decode1)
        left_decode2 = self.skip2(left_encode0, left_decode2)
        right_decode2 = self.skip2(right_encode0, right_decode2)
        right_decode2 = self.frequency_transfer1(base = right_decode2, ref = left_decode2)
        #-----------
        left_decode2 = self.fus2(left_decode2,left_encode2)
        right_decode2 = self.fus2(right_decode2,right_encode2)
        left_decode3 = self.layer_up1(left_decode2)
        right_decode3 = self.layer_up1(right_decode2)
        left_decode3 = self.skip3(left_encode0, left_decode3)
        right_decode3 = self.skip3(right_encode0, right_decode3)
        left_decode3 = self.color_transfer1(base = left_decode3, ref = right_decode3)
        #-----------
        left_decode3 = self.fus3(left_decode3,left_encode3)
        right_decode3 = self.fus3(right_decode3,right_encode3)
        #==============================output==============================
        left_fea0 = left_encode3
        right_fea0 = right_encode3   
        left_fea1 = left_encode0
        right_fea1 = right_encode0
        left_fea2 = left_decode3
        right_fea2 = right_decode3
        #===================pre-reconstruction constraint===================
        left_pre= self.conv_out1(left_decode3)
        right_pre = self.conv_out1(right_decode3)
        return left_fea0, right_fea0, left_fea1, right_fea1, left_fea2, right_fea2, left_pre, right_pre

class Basicblock(nn.Module):
    def __init__(self,channels,mode,dil):
        super(Basicblock, self).__init__()
        if mode == "up":
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
            )
        elif mode == "down":
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
            )
        elif mode == "no":
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 1, 1, 0),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
            )
        self.body = nn.Sequential(
            HinBlock(channels, channels),
        )

    def forward(self, x):
        return self.conv(self.body(x))

class SkipFusion(nn.Module):
    def __init__(self,channels):
        super(SkipFusion, self).__init__()
        self.fus = nn.Sequential(
            HinBlock(channels*2, channels),
        )
        self.alpha1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha1.data.fill_(1.0)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha2.data.fill_(0.5)

    def forward(self, x, y):
        out = self.alpha1 * self.fus(torch.cat([x,y],1)) + self.alpha2 * x
        return out

class SkipConnection(nn.Module):
    def __init__(self,channels):
        super(SkipConnection, self).__init__()
        self.fus = nn.Sequential(
                            nn.Conv2d(channels*2, channels, 1, 1, 0, bias=True),
                            nn.InstanceNorm2d(channels, affine=True),
                            nn.ReLU(inplace = True),
                            )
        self.alpha1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha1.data.fill_(1.0)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha2.data.fill_(0.5)

    def forward(self, x, y):
        out = self.alpha1 * self.fus(torch.cat([x,y],1)) + self.alpha2 * x
        return out

class SFM_block(nn.Module):
    def __init__(self, channels):
        super(SFM_block, self).__init__()
        self.modulate1 = AdaIN(channels)
        self.modulate2 = AdaIN(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, 1, 0)
        )

    def fft(self, x):
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        return mag, pha

    def ifft(self, mag, pha, H, W):
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        fre_out = torch.complex(real, imag)
        x = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return x

    def forward(self, base, ref):
        b,c,h,w = base.shape
        base_mag, base_pha = self.fft(base)
        ref_mag, ref_pha = self.fft(ref)
        new_mag = self.modulate1(base = base_mag, ref = ref_mag)
        new_pha = self.modulate2(base = base_pha, ref = ref_pha)
        new = self.ifft(new_mag, new_pha, h, w)
        new = self.conv(torch.cat([new,base],1))
        return new

class AdaIN(nn.Module):
    def __init__(self, channels):
        super(AdaIN, self).__init__()
        self.conv_ref = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(inplace = True),
        )
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, base, ref):
        b,c,h,w = base.shape
        base = self.norm(base)
        ref = self.conv_ref(ref)
        #
        ref = self.pool(ref)
        gamma = self.conv_gamma(ref).expand_as(base)
        beta = self.conv_beta(ref).expand_as(base)
        #
        return base * (gamma + 1) + beta

class SCM_block(nn.Module):
    def __init__(self, channels):
        super(SCM_block, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv_fea1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                           nn.GroupNorm(num_groups=16, num_channels=channels))
        self.conv_fea2 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                           nn.GroupNorm(num_groups=16, num_channels=channels))
        self.conv_info1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                           nn.GroupNorm(num_groups=16, num_channels=channels))
        self.conv_info2 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                           nn.GroupNorm(num_groups=16, num_channels=channels))
        self.fus = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                           nn.GroupNorm(num_groups=16, num_channels=channels))
        self.out = nn.Sequential(ResBlock(channels*2, channels, channels))
        self.pool = nn.MaxPool2d(4,2,1)

    def forward(self, base, ref):
        base_info = self.conv_info2(base)
        base = self.pool(base)
        ref = self.pool(ref)
        b, c, h, w = base.shape
        ref_info = self.conv_info1(ref)
        ref_fea = self.conv_fea1(ref)
        base_fea = self.conv_fea2(base) 
        # PAM
        base_fea = rearrange(base_fea, "B C H W -> (B H) W C")
        ref_fea = rearrange(ref_fea, "B C H W -> (B H) C W")
        score = torch.bmm(base_fea, ref_fea)
        M = self.softmax(score) # (B H) W W
        #warp
        ref_info = rearrange(ref_info, "B C H W -> (B H) W C")
        warp_info = torch.bmm(M, ref_info)
        warp_info = rearrange(warp_info, "(B H) W C -> B C H W", H = h)
        warp_info = nn.functional.interpolate(warp_info, scale_factor=2, mode='bicubic', align_corners=True)
        warp_info = self.fus(warp_info)
        #fusion
        new = self.out(torch.cat([warp_info, base_info],1))
        return new