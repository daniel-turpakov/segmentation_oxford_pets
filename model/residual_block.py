import torch
from torch import nn
from torchinfo import summary

from model.cbam import ChannelAttention, SpatialAttention, CBAMSkip
from model.basic_unet import UNet
from model.mbconv import MBConv


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv_block(x))

class ResBottleneck(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()

        self.res_block1 = ResBlock(channels)
        self.res_block2 = ResBlock(channels)

        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.relu(x + residual)


if __name__ == '__main__':
    test_tensor = torch.randn(1, 3, 512, 512)
    model = UNet(3, 1, [32, 64, 128, 256],
                 conv_block=MBConv,
                 bottleneck_block=ResBottleneck,
                 skip_block=CBAMSkip)

    assert model(test_tensor).size() == torch.Size(
        [1, 1, 512, 512]), "Incorrect output size, check the implementation for potential errors"

    summary(model, input_data=test_tensor, device=torch.device("cpu"))
