import torch
import torch.nn as nn
from model_utils import VGGBlock, UpConv


class UnetPP(nn.Module):
    def __init__(self, input_channels=6) -> None:
        super().__init__()

        n_filter = [32, 64, 128]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(input_channels, n_filter[0], n_filter[0])
        self.conv1_0 = VGGBlock(n_filter[0], n_filter[1], n_filter[1])
        self.conv2_0 = VGGBlock(n_filter[1], n_filter[2], n_filter[2])

        self.convT0_0 = UpConv(n_filter[1], n_filter[1])
        self.convT2_0 = UpConv(n_filter[2], n_filter[2])

        self.conv0_1 = VGGBlock(n_filter[0] + n_filter[1], n_filter[0], n_filter[0])
        self.conv1_1 = VGGBlock(n_filter[1] + n_filter[2], n_filter[1], n_filter[1])

        self.convT1_1 = UpConv(n_filter[1], n_filter[1])

        self.conv0_2 = VGGBlock(n_filter[0] * 2 + n_filter[1], n_filter[0], n_filter[0])

        self.Conv_1x1 = nn.Conv2d(n_filter[0], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.convT0_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.convT2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.convT1_1(x1_1)], 1))

        out = self.Conv_1x1(x0_2)

        return torch.sigmoid(out)
