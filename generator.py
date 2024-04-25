import torch
import torch.nn as nn
from torch.nn import functional

# Define the Generator Network
class Generator(nn.Module):
    """
    Generator for the DCGAN.
    Input tensor x of shape [batch_size, nz, 1, 1] 
    Outputs a generated image tensor of shape [batch_size, nc, image_size, image_size].

    Args:
        params (dict): A dictionary containing the network hyperparameters. 
                       - 'nz': input noise vector
                       - 'ngf': number of generator filters
                       - 'nc': number of output channels
    """
    def __init__(self, params):
        super().__init__()

        self.conv2 = nn.ConvTranspose2d(
            in_channels=params['ngf']*8,
            out_channels=params['ngf']*4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        self.conv3 = nn.ConvTranspose2d(
            in_channels=params['ngf']*4,
            out_channels=params['ngf']*2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        self.conv4 = nn.ConvTranspose2d(
            in_channels=params['ngf']*2,
            out_channels=params['ngf']*1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        self.conv5 = nn.ConvTranspose2d(
            in_channels=params['ngf'],
            out_channels=params['nc'],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )


    def forward(self, x):
        """
        Forward pass of the generator network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, params['nz'], 1, 1] 
                              - 'nz': input noise vector

        Returns:
            torch.Tensor: [batch_size, params['nc'], params['image_size'], params['image_size']].
                          - 'nc': number of output channels
        """
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        x = functional.relu(self.bn4(self.conv4(x)))

        x = functional.tanh(self.conv5(x))

        return x
