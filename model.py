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


class Minibatch_Discrimination(nn.Module):
    def __init__(self, batch_size, n_feat, hidden_dim=20, resolution=(32, 32)):
        """
        Minibatch discrimination layer as described in Improved Techniques...
        This function only works with fixed batch size.
        Args:
            batch_size (int): batch size in training
            n_feat (int): number of input feature maps
            hidden_dim (int): hidden dimension of tensor T
            resolution tuple(int, int): height and width of the input feature
            maps
        """
        super(Minibatch_Discrimination, self).__init__()
        self.batch_size = batch_size
        self.n_feat = n_feat
        self.resolution = resolution
        self.hidden_dim = hidden_dim

        tensor_data = torch.randn((n_feat * resolution[0] * resolution[1],
                                   self.hidden_dim * resolution[0] * resolution[1]))
        self.tensor = nn.Parameter(data=tensor_data, requires_grad=True)

    def forward(self, x):
        # reshape data from (N, C, H, W) to (N, (C*H*W))
        x = x.view(self.batch_size, -1)
        x = torch.matmul(x, self.tensor)
        x = x.view(self.batch_size, -1, self.hidden_dim)
        distances = [[] for i in range(self.batch_size)]

        #TODO: simplify this part. Is there a better way than using for-loops?
        for i in range(self.batch_size):
            for j in range(i + 1, self.batch_size):
                l1_matrix = torch.linalg.vector_norm(x[i] - x[j], ord=1, dim=1)
                distance = torch.exp(-l1_matrix)
                distances[i].append(distance)

        for i in range(1, self.batch_size):
            for j in range(i):
                distances[i].append(distances[j][0])

        results = []
        for i in range(self.batch_size):
            result = torch.sum(torch.stack(distances[i]), dim=0)
            result = result.view(1, *self.resolution)
            results.append(result)

        results = torch.stack(results, dim=0)

        return results


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


class Discriminator_MD(nn.Module):
    """
    Discriminator with minibatch discrimination.
    """
    def __init__(self, batch_size, n_feat=64):
        """
        Args:
            n_feat (int): number of input feature maps in the first layer.
        """
        super(Discriminator_MD, self).__init__()
        self.n_feat = n_feat

        self.layer1 = Conv_BN_LReLU(3, self.n_feat, 5, 2, 2, 1)
        #self.md1 = Minibatch_Discrimination(batch_size, self.n_feat, resolution=(32, 32))

        self.layer2 = Conv_BN_LReLU(self.n_feat, self.n_feat*2, 5, 2, 2, 1)
        self.md2 = Minibatch_Discrimination(batch_size, self.n_feat*2, resolution=(16, 16))

        self.layer3 = Conv_BN_LReLU(self.n_feat*2+1, self.n_feat*4, 5, 2, 2, 1)
        self.md3 = Minibatch_Discrimination(batch_size, self.n_feat*4, resolution=(8, 8))

        self.layer4 = Conv_BN_LReLU(self.n_feat*4+1, self.n_feat*8, 5, 2, 2, 1)
        self.md4 = Minibatch_Discrimination(batch_size, self.n_feat*8, resolution=(4, 4))

        self.out = nn.Sequential(
            nn.Conv2d(self.n_feat*8+1, 1, 4, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        #x1_md = self.md(x1)
        #x1 = torch.cat([x1, x1_md], dim=1)

        x2 = self.layer2(x1)
        x2_md = self.md2(x2)
        x2 = torch.cat([x2, x2_md], dim=1)

        x3 = self.layer3(x2)
        x3_md = self.md3(x3)
        x3 = torch.cat([x3, x3_md], dim=1)

        x4 = self.layer4(x3)
        x4_md = self.md4(x4)
        x4 = torch.cat([x4, x4_md], dim=1)

        out = torch.squeeze(self.out(x4))

        return out, [x1, x2, x3, x4]  # for feature matching
