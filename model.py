import torch
from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-level U-Net with 16 base channels, trained from scratch."""
    def __init__(self):
        super().__init__()
        widths = (16, 32, 64, 128, 256)
        self.down = nn.ModuleList(conv_block(a, b) for a, b in zip((1,) + widths, widths))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2)
                                for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.decode = nn.ModuleList(conv_block(2*b, b) for b in widths[-2::-1])
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        skips = []
        for layer in self.down[:-1]:
            x = layer(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.down[-1](x)
        for up, decode, skip in zip(self.up, self.decode, reversed(skips)):
            x = decode(torch.cat((up(x), skip), dim=1))
        return self.output(x)
