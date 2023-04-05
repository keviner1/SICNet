import torch
import torch.nn as nn
from einops import rearrange
from .ResBlock import ResBlock,HinBlock

class Colorization(nn.Module):
    def __init__(self, channels):
        super(Colorization, self).__init__()
        #==============================PCT-block==============================
        self.softmax = nn.Softmax(dim=2)
        self.conv_fea = nn.Sequential(HinBlock(channels*2, channels*2),)
        self.convL_1 = nn.Sequential(nn.Conv2d(channels*2, channels*2, 1, 1, 0, bias=True),
                                         nn.GroupNorm(16, channels*2))
        self.convR_1 = nn.Sequential(nn.Conv2d(channels*2, channels*2, 1, 1, 0, bias=True),
                                         nn.GroupNorm(16, channels*2))
        self.convR_2 = nn.Sequential(ResBlock(channels*2, channels, channels))
        self.convL_2 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                         nn.GroupNorm(16, channels))
        self.convL_3 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                                        nn.GroupNorm(16, channels))
        #==============================Refine===============================
        self.Resblocks_out = nn.Sequential(
                HinBlock(channels*2+1, channels),
                HinBlock(channels, channels),
                HinBlock(channels, channels),
            )
        self.conv_out = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1,bias=False),
                                      nn.GroupNorm(8, channels),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, 3, 1, 1, 0, bias=False))

    def forward(self, left0, right0, left1, right1, left2, right2, left_mono, left_pre, TR):
        b, c, h, w = left0.shape
        #==============================PCT-block==============================
        left_fea = torch.cat([left1, left2],1)
        right_fea = torch.cat([right1, right2],1)
        left_fea = self.convL_1(self.conv_fea(left_fea))
        right_fea = self.convR_1(self.conv_fea(right_fea))
        left_fea = rearrange(left_fea, "B C H W -> (B H) W C")
        right_fea = rearrange(right_fea, "B C H W -> (B H) C W")
        score = torch.bmm(left_fea, right_fea)
        M = self.softmax(score) # (B H) W W
        right_color = torch.cat([right0, right2],1)
        right_color = self.convR_2(right_color)
        right_color = rearrange(right_color, "B C H W -> (B H) W C")
        left_color = torch.bmm(M, right_color)
        left_color = rearrange(left_color, "(B H) W C -> B C H W", H = h)
        left_color = self.convL_3(left_color)
        left_shortcut = self.convL_2(left0)
        #==============================Refine===============================
        output = self.Resblocks_out( torch.cat([left_mono, left_shortcut, left_color],1) )
        output = self.conv_out(output)
        
        #directly warping for visualizing the established stereo correspondence
        # TR = rearrange(TR, "B C H W -> (B H) W C")
        # left_color = torch.bmm(M, TR)
        # left_color = rearrange(left_color, "(B H) W C -> B C H W", H = h)
        # output = left_color

        return M, output