import torch
import torch.nn as nn
from einops import rearrange

class Unet(nn.Module):
    def __init__(self, channels):
        super(Unet, self).__init__()
        self.layer_dowm1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 2, dilation = 2),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),
        )
        self.layer_dowm2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 4, dilation = 4),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),
        )
        self.layer_dowm3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 8, dilation = 8),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),
        )
        self.layer0 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            )
        self.layer_up1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels, 3, 1, 2, dilation = 2),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),       
        ) 
        self.layer_up2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels, 3, 1, 4, dilation = 4),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),
        )
        self.layer_up3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels, 3, 1, 8, dilation = 8),nn.GroupNorm(16, channels),nn.ReLU(inplace=True),
        )        
        self.fus1 = SkipFusion(channels)
        self.fus2 = SkipFusion(channels)
        self.fus3 = SkipFusion(channels)
        self.conv_in = nn.Sequential(nn.Conv2d(3, channels, 1, 1, 0), nn.GroupNorm(16,channels), nn.ReLU(inplace=True))


    def forward(self, x):
        x_encode3 = self.conv_in(x)
       #down
        x_encode2 = self.layer_dowm1(x_encode3)
        x_encode1 = self.layer_dowm2(x_encode2)
        x_encode0 = self.layer_dowm3(x_encode1)
        #----
        x_encode0 = self.layer0(x_encode0)
        #x
        x_decode1 = self.layer_up3(x_encode0)
        x_decode1 = self.fus1(x_decode1,x_encode1)

        x_decode2 = self.layer_up2(x_decode1)
        x_decode2 = self.fus2(x_decode2,x_encode2)

        x_decode3 = self.layer_up1(x_decode2)
        x_decode3 = self.fus3(x_decode3,x_encode3)

        return x_decode3

class SkipFusion(nn.Module):
    def __init__(self,channels):
        super(SkipFusion, self).__init__()
        self.fus = nn.Sequential(
            nn.Conv2d(channels*2, channels, 3, 1, 1), nn.GroupNorm(16, channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        out = self.fus(torch.cat([x,y],1))
        return out
