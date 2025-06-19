from typing import Optional, Any
from torchinfo import summary

from model.basic_blocks import *


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filters: list[int],
                 conv_block: Optional[nn.Module] = None,
                 bottleneck_block: Optional[nn.Module] = None,
                 downsampling_block: Optional[nn.Module] = None,
                 upsampling_block: Optional[nn.Module] = None,
                 skip_block: Optional[nn.Module] = None,
                 block_kwargs: dict[str, dict[str, Any]] = {}):
        super().__init__()

        self.n_blocks = len(filters)
        self.blocks = {
            'conv': conv_block or BasicBlock,
            'bottleneck': bottleneck_block or BasicBottleneck,
            'down': downsampling_block or BasicDown,
            'up': upsampling_block or BasicUp,
            'skip': skip_block or BasicSkip
        }
        self.kwargs = {
            'conv': block_kwargs.get('conv', {}),
            'bottleneck': block_kwargs.get('bottleneck', {}),
            'down': block_kwargs.get('down', {}),
            'up': block_kwargs.get('up', {}),
            'skip': block_kwargs.get('skip', {})
        }

        self.conv2d_in = nn.Conv2d(in_channels, filters[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            self.blocks['conv'](filters[i-1], **self.kwargs['conv'])
            for i in range(1, self.n_blocks)
        ])

        self.downsampling_blocks = nn.ModuleList([
            self.blocks['down'](filters[i-1], filters[i], **self.kwargs['down'])
            for i in range(1, self.n_blocks)
        ])

        # Bottleneck
        self.bottleneck = self.blocks['bottleneck'](filters[-1], **self.kwargs['bottleneck'])

        # Decoder
        self.upsampling_blocks = nn.ModuleList([
            self.blocks['up'](filters[i], filters[i-1], **self.kwargs['up'])
            for i in range(self.n_blocks-1, 0, -1)
        ])

        self.skip_blocks = nn.ModuleList([
            self.blocks['skip'](filters[i-1], **self.kwargs['skip'])
            for i in range(self.n_blocks-1, 0, -1)
        ])

        self.decoder_blocks = nn.ModuleList([
            self.blocks['conv'](filters[i-1], **self.kwargs['conv'])
            for i in range(self.n_blocks-1, 0, -1)
        ])

        self.conv2d_out = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_in(x)

        # Encoder path with skip connections
        skips = []
        for encode, down in zip(self.encoder_blocks, self.downsampling_blocks):
            x = encode(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        # Decoder path
        for up, skip_connection, decode in zip(
            self.upsampling_blocks,
            self.skip_blocks,
            self.decoder_blocks
        ):
            x = up(x)
            x = skip_connection(x, skips.pop())
            x = decode(x)

        return self.conv2d_out(x)


if __name__ == '__main__':
    test_tensor = torch.randn(1, 3, 512, 512)
    model = UNet(3, 1, [16, 32, 64, 128])

    assert model(test_tensor).size() == torch.Size(
        [1, 1, 512, 512]), "Incorrect output size, check the implementation for potential errors"

    summary(model, input_data=test_tensor, device=torch.device("cpu"))
