import torch
import torch.nn as nn
from einops import rearrange
from .ResBlock import ResBlock

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.conv_in = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),nn.GroupNorm(16, channels))
        self.conv_fus = nn.Sequential(nn.Conv2d(4, channels, 3, 1, 1, bias=True),nn.GroupNorm(16, channels))
        self.Resblocks = nn.Sequential(
                ResBlock(channels, channels, channels),
                ResBlock(channels, channels, channels),
        )
        self.out = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1,bias=False),
                nn.GroupNorm(8, channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channels, 3, 1, 1, 0, bias=False)
            )

    

    def forward(self, left_fea, right_fea, left_gray, left_color, right_color, mode):
        b,c,h,w = left_fea.shape
        left_fea = self.conv_in(left_fea)
        right_fea = self.conv_in(right_fea)
        left_fea = rearrange(left_fea, "B C H W -> (B H) W C")
        right_fea = rearrange(right_fea, "B C H W -> (B H) C W")
        score = torch.bmm(left_fea, right_fea) 
        M = self.softmax(score) # (B H) W W

        #
        if mode == "teacher":
            left_fus = 0
        elif mode == "direct":
            left_fus = 0
            left_color = rearrange(left_color, "B C H W -> (B H) W C")
            right_color = rearrange(right_color, "B C H W -> (B H) C W")
            score = torch.bmm(left_color, right_color) 
            M = self.softmax(score) # (B H) W W
        else:
            right_color = rearrange(right_color, "B C H W -> (B H) W C")
            warp_color = torch.bmm(M, right_color)
            warp_color  = rearrange(warp_color, "(B H) W C -> B C H W", H = h)
            left_fus = torch.cat([left_gray,warp_color],1)
            #
            left_fus = self.conv_fus(left_fus)
            left_fus = self.Resblocks(left_fus)
            left_fus = self.out(left_fus)

        return M, left_fus
