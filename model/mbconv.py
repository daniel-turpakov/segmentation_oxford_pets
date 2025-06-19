import torch
from torch import nn
from typing import Union
from torchinfo import summary

from model.basic_unet import UNet


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_rate: int) -> None:
        super().__init__()

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // squeeze_rate, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // squeeze_rate, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se_block(x)
        return scale * x



class MBConv(nn.Module):
    def __init__(self, in_channels: int, exp_channels: int = 0, kernel_size: Union[int, tuple[int, int]] = 3,
                 padding: Union[int, tuple[int, int]] = 1, stride: Union[int, tuple[int, int]] = 1, non_linearity: str = 'RE',
                 se_block: bool = True, squeeze_rate: int = 16) -> None:
        super().__init__()

        self.non_linearity_bank = {'RE': nn.ReLU6, 'HS': nn.Hardswish}

        out_channels = in_channels

        self.use_skip_connection = stride != 2 and in_channels == out_channels

        self.layers = []

        if not exp_channels:
            exp_channels = 4 * in_channels

        if exp_channels != in_channels:
            self.layers.append(nn.Conv2d(in_channels, exp_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(exp_channels))
            self.layers.append(self.non_linearity_bank[non_linearity]())

        self.layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels))
        self.layers.append(nn.BatchNorm2d(exp_channels))
        self.layers.append(self.non_linearity_bank[non_linearity]())

        if se_block:
            self.layers.append(SqueezeExcitation(exp_channels, squeeze_rate))

        self.layers.append(nn.Conv2d(exp_channels, out_channels, kernel_size=1))
        self.layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        # Если пространственный размер входного тензора не меняется, то прибавляем скип
        if self.use_skip_connection:
            out = out + x
        return out


if __name__ == '__main__':
    test_tensor = torch.randn(1, 3, 512, 512)
    model = UNet(3, 1, [16, 32, 64, 128],
                 conv_block=MBConv)

    assert model(test_tensor).size() == torch.Size(
        [1, 1, 512, 512]), "Incorrect output size, check the implementation for potential errors"

    summary(model, input_data=test_tensor, device=torch.device("cpu"))
