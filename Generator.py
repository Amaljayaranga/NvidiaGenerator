import torch.nn as nn
import torch


Z_SIZE = 256


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)
        return x * tmp

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.view(Z_SIZE,1,1,1)
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(PixelNorm(),
                                     nn.Linear(Z_SIZE, Z_SIZE),
                                     nn.LeakyReLU(),
                                     PixelNorm(),
                                     Reshape(),
                                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(),
                                     PixelNorm()
                                     )

    def forward(self, z):
        x = self.initial(z)
        print("forward", x.shape)



z = torch.randn(1,256)
gen = Generator()
v = gen(z)
