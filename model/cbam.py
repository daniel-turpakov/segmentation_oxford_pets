import torch
from torch import nn
from torchinfo import summary

from model.basic_unet import UNet
from model.mbconv import MBConv


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAMSkip(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(2 * channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return self.conv(x)



if __name__ == '__main__':
    test_tensor = torch.randn(1, 3, 512, 512)
    model = UNet(3, 1, [16, 32, 64, 128],
                 conv_block=MBConv,
                 skip_block=CBAMSkip)

    assert model(test_tensor).size() == torch.Size(
        [1, 1, 512, 512]), "Incorrect output size, check the implementation for potential errors"

    summary(model, input_data=test_tensor, device=torch.device("cpu"))
