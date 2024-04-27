import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.ops.misc import ConvNormActivation


def create_mobilenet():
    MODEL = deeplabv3_mobilenet_v3_large()
    MODEL.backbone._modules['0'] =  ConvNormActivation(16, 16, 3, 2)

    MODEL.classifier[4] = nn.Conv2d(256, 1, (1, 1), (1, 1))

    return MODEL
