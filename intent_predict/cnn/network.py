import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms

import numpy as np
from torchvision.models import inception

class IntentNet(nn.Module):
    """
    The intent net structure
    """
    def __init__(self):
        """
        Instantiate the model
        """
        super(IntentNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        # Keep all convolutional layers and disgard the final avgpool and fc
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))

        # Lock the weights for the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3),
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=100*100, out_features=20),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=20, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        forward method of the network
        """
        x = self.backbone(inputs)

        x1 = self.deConv(x)

        x2 = torch.flatten(x1, 1)
        x2 = self.fc(x2)

        return x1, x2