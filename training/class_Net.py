import torch.nn as nn
from torchvision import models


def create_resnet18_model():
    MODEL = models.resnet18()
    num_ftrs = MODEL.fc.in_features
    MODEL.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
    MODEL.fc = nn.Linear(num_ftrs, 1)

    return MODEL
