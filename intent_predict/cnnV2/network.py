import torch
import torch.nn as nn
import numpy as np

from torchvision import models

from torchvision.models import inception


class SimpleCNN(nn.Module):
    """
    Simple CNN.
    """
    def __init__(self, dropout_p = 0.2):
        """
        Instantiate the model
        """
        super(SimpleCNN, self).__init__()

        self.img_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=20)
        )

        self.img_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=15, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=15)
        )
        
        self.img_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=10)
        )

        self.img_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=5)
        )
        
        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
        )
        
        IMG_LAYER_OUTPUT_SIZE = 196020
        NON_SPATIAL_FEATURE_SIZE = 5
        
        
        self.linear_layer1 = nn.Sequential(
            nn.Linear(IMG_LAYER_OUTPUT_SIZE + NON_SPATIAL_FEATURE_SIZE, 1000),
            nn.ReLU(inplace=True),
        )
        
        self.linear_layer2 = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
        )
        
        self.linear_layer3 = nn.Sequential(
            nn.Linear(100, 1),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feature, non_spatial_feature):
        """
        forward method
        """
        x = self.img_layer1(img_feature)
        x = self.dropout(x)

        x = self.img_layer2(x)
        x = self.dropout(x)

        x = self.img_layer3(x)
        x = self.dropout(x)
        
        x = self.img_layer4(x)
        x = self.dropout(x)
        
        x = self.flatten_layer(x)
        x = self.dropout(x)
        
        
        non_spatial_feature = self.flatten_layer(non_spatial_feature)
        
        x = torch.cat([x, non_spatial_feature], 1)
        x = self.linear_layer1(x)
        x = self.dropout(x)

        x = self.linear_layer2(x)
        x = self.dropout(x)

        x = self.linear_layer3(x)
        x = self.dropout(x)

        x = self.sigmoid(x)

        return x