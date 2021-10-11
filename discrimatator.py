import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.utils.spectral_norm as spectral_norm

torch.manual_seed(1)


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        # x = F.relu(x)
        # x = F.leaky_relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.conv0 = spectral_norm(nn.Conv2d(6, 16, 3, 1, padding=(1, 1)))
        self.conv1 = spectral_norm(nn.Conv2d(16, 32, 3, 1, padding=(2, 2), dilation=2))
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, 3, 1, padding=(3, 3), dilation=3))
        self.conv3 = spectral_norm(nn.Conv2d(64, 128, 3, 1, padding=(4, 4), dilation=4))
        self.conv4 = spectral_norm(nn.Conv2d(64, 1, 3, 1, padding=(1, 1)))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        # self.relu3 = TReLU()
        self.relu4 = TReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu4(x)
        res = self.conv4(x)
        # res = F.sigmoid(res)
        return res
