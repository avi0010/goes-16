import torch
import torch.nn as nn
from model_utils import Attention_block, RRCNN_block, UpConv


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=6, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        n_filter = [32, 64, 128]

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=n_filter[0], t=t)

        self.RRCNN2 = RRCNN_block(ch_in=n_filter[0], ch_out=n_filter[1], t=t)

        self.RRCNN3 = RRCNN_block(ch_in=n_filter[1], ch_out=n_filter[2], t=t)

        self.Up3 = UpConv(ch_in=n_filter[2], ch_out=n_filter[1])
        self.Att3 = Attention_block(F_g=n_filter[1], F_l=n_filter[1], F_int=16)
        self.Up_RRCNN3 = RRCNN_block(ch_in=n_filter[2], ch_out=n_filter[1], t=t)

        self.Up2 = UpConv(ch_in=n_filter[1], ch_out=n_filter[0])
        self.Att2 = Attention_block(F_g=n_filter[0], F_l=n_filter[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=n_filter[1], ch_out=n_filter[0], t=t)

        self.Conv_1x1 = nn.Conv2d(
            n_filter[0], output_ch, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        # decoding + concat path
        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
