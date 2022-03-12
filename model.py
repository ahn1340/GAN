import torch
from torch import nn


class ConvT_BN_ReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        super(ConvT_BN_ReLU, self).__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.op(x)

class Conv_BN_LReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        super(Conv_BN_LReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.op(x)

class Generator(nn.Module):
    """
    Convolutional generator module with architecture specified in the DCGAN
    paper.
    """
    def __init__(self, z_dim=64, n_feat=64):
        """
        Args:
            z_dim (int): dimension of z (random noise).
            n_feat (int): number of input feature maps in the final layer.
            The feature map sizes of earlier layers are computed according
            to this value.
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.n_feat = n_feat  # number of feature maps in the final layer

        self.layer1 = ConvT_BN_ReLU(self.z_dim, self.n_feat * 8, 4, 1, 0, 1)
        self.layer2 = ConvT_BN_ReLU(self.n_feat * 8, self.n_feat * 4, 4, 2, 1, 1)
        self.layer3 = ConvT_BN_ReLU(self.n_feat * 4, self.n_feat * 2, 4, 2, 1, 1)
        self.layer4 = ConvT_BN_ReLU(self.n_feat * 2, self.n_feat, 4, 2, 1, 1)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.n_feat, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # reshape z: (N, C) to (N, C, 1, 1).
        z = z.view(-1, self.z_dim, 1, 1)
        z1 = self.layer1(z)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        image = self.out(z4)

        return image#, [z1, z2, z3, z4]


class Discriminator(nn.Module):
    """
    Convolutional discriminator module with architecture specified in the
    DCGAN paper.
    """
    def __init__(self, n_feat=64):
        """
        Args:
            n_feat (int): number of input feature maps in the first layer.
        """
        super(Discriminator, self).__init__()
        self.n_feat = n_feat

        self.layer1 = Conv_BN_LReLU(3, self.n_feat, 5, 2, 2, 1)
        self.layer2 = Conv_BN_LReLU(self.n_feat, self.n_feat*2, 5, 2, 2, 1)
        self.layer3 = Conv_BN_LReLU(self.n_feat*2, self.n_feat*4, 5, 2, 2, 1)
        self.layer4 = Conv_BN_LReLU(self.n_feat*4, self.n_feat*8, 5, 2, 2, 1)
        self.out = nn.Sequential(
            nn.Conv2d(self.n_feat*8, 1, 4, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = torch.squeeze(self.out(x4))

        return out, [x1, x2, x3, x4]  # for feature matching