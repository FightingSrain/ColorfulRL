import torch.nn.functional as F
import numpy as np
from .unet_part import *

class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel=32):
    super().__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, [3, 3], padding=1)
    self.bn = nn.BatchNorm2d(out_channel)
    self.conv1 = nn.Conv2d(out_channel, out_channel, [3, 3], padding=1)
    self.leaky_relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1((self.leaky_relu(self.conv1(inputs))))
    return x + inputs

class ModUnet(nn.Module):
    def __init__(self,  config, bilinear=True):
        super(ModUnet, self).__init__()
        self.n_channels = 3
        self.n_classes = config.num_actions
        self.bilinear = bilinear
        num_blocks = 4

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.resblock = nn.Sequential(*[ResBlock(1024 // factor, 1024 // factor) for i in range(num_blocks)])
        self.up1_a = Up(1024, 512 // factor, bilinear)
        self.up2_a = Up(512, 256 // factor, bilinear)
        self.up3_a = Up(256, 128 // factor, bilinear)
        self.up4_a = Up(128, 64, bilinear)
        self.outc_pi = OutConv(64, config.num_actions)

        self.up1_c = Up(1024, 512 // factor, bilinear)
        self.up2_c = Up(512, 256 // factor, bilinear)
        self.up3_c = Up(256, 128 // factor, bilinear)
        self.up4_c = Up(128, 64, bilinear)
        self.outc_mean = OutConv(64, config.num_actions)
        self.outc_logstd = nn.Parameter(torch.zeros(1, config.num_actions), requires_grad=True)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p
    def forward(self, x, batch_size, action_num):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xa = self.up1_a(x5, x4)
        xa = self.up2_a(xa, x3)
        xa = self.up3_a(xa, x2)
        xa = self.up4_a(xa, x1)
        policy = F.softmax(self.outc_pi(xa), 1)

        xc = self.up1_c(x5, x4)
        xc = self.up2_c(xc, x3)
        xc = self.up3_c(xc, x2)
        xc = self.up4_c(xc, x1)
        mean = self.parse_p((self.outc_mean(xc))).view(x.shape[0], action_num, 1, 1)
        # mean = self.outc_mean(xc)
        logstd = self.outc_logstd.expand([batch_size, action_num]).view(x.shape[0], action_num, 1, 1)


        xv = self.up1(x5, x4)
        xv = self.up2(xv, x3)
        xv = self.up3(xv, x2)
        xv = self.up4(xv, x1)
        value = self.outc(xv)

        return policy, value, mean, logstd
