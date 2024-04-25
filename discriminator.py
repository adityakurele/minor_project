import torch
import torch.nn as nn
from torch.nn import functional as F

# Define the Discriminator Network
# class Discriminator(nn.Module):
#     def __init__(self,params):
#         """
#         Args:
#         - nc : Number of Input channels
#         - ndf: Number of discriminator filters
#         """
#         super().__init__()

#         self.conv1 = nn.Conv2d(
#             in_channels=params['nc'], 
#             out_channels=params['ndf'], 
#             kernel_size=4, 
#             stride=2, 
#             padding=1, 
#             bias=False)

#         self.conv2 = nn.Conv2d(
#             in_channels=params['ndf'], 
#             out_channels=params['ndf'] * 2, 
#             kernel_size=4, 
#             stride=2, 
#             padding=1, 
#             bias=False)
        
#         self.bn2 = nn.BatchNorm2d(num_features=params['ndf'] * 2)

#         self.conv3 = nn.Conv2d(
#             in_channels=params['ndf'] * 2, 
#             out_channels=params['ndf'] * 4, 
#             kernel_size=4, 
#             stride=2, 
#             padding=1, 
#             bias=False)
        
#         self.bn3 = nn.BatchNorm2d(num_features=params['ndf'] * 4)

#         self.conv4 = nn.Conv2d(
#             in_channels=params['ndf'] * 4, 
#             out_channels=params['ndf'] * 8, 
#             kernel_size=4, 
#             stride=2, 
#             padding=1, 
#             bias=False)
        
#         self.bn4 = nn.BatchNorm2d(num_features=params['ndf'] * 8)

#         self.conv5 = nn.Conv2d(
#             in_channels=params['ndf'] * 8, 
#             out_channels=1, 
#             kernel_size=4, 
#             stride=1, 
#             padding=0, 
#             bias=False)

#     def forward(self, x):
#         """
#         Returns:
#         - out (torch.Tensor): tensor of probabability(real)
#         """
#         x = functional.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
#         x = functional.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2, inplace=True)
#         x = functional.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2, inplace=True)
#         x = functional.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2, inplace=True)

#         out = torch.sigmoid(self.conv5(x))

#         return out

class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.sigmoid(self.conv5(x))

        return x