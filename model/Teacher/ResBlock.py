import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, inner_channel, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.inner_channel = inner_channel
        self.out_channels = out_channels

        self.mainstream = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channel, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channel, self.inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channel, self.out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.out_channels),
            )
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mainstream(x)
        x = self.shortcut(x)
        out = out + x
        # out = self.act(out)
        return out