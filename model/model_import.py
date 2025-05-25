from ultralytics import YOLO
import torch
import cv2
from torchvision import transforms

import torch.nn as nn
import torchvision.models as models


class ResNet18Regressor(nn.Module):
    def __init__(self, use_mask=False):
        super(ResNet18Regressor, self).__init__()
        in_channels = 4 if use_mask else 3

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if in_channels != 3:
            original_first_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels,
                original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias
            )

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.resnet(x)


def load_reg_model():
    model_reg = ResNet18Regressor(use_mask=False)
    model_reg = torch.load('model/Food_estimator.pt', map_location='cuda')
    return model_reg

def load_seg_model():
    model = YOLO("model/Food_seg_model2.pt")
    return model
