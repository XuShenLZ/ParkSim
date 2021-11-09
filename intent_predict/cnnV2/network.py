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
        
        self.image_layers = []

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.BatchNorm2d(num_features=20)
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.BatchNorm2d(num_features=15)
        ))
        
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.BatchNorm2d(num_features=15)
        ))
        
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.BatchNorm2d(num_features=15)
        ))
        
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.BatchNorm2d(num_features=15)
        ))
        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
        )
        
        IMG_LAYER_OUTPUT_SIZE = 196020
        NON_SPATIAL_FEATURE_SIZE = 5
        
        
        self.linear_layer1 = nn.Sequential(
            nn.Linear(IMG_LAYER_OUTPUT_SIZE + NON_SPATIAL_FEATURE_SIZE, 1000),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(1000, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feature, non_spatial_feature):
        """
        forward method
        """
        x = img_feature
        
        for img_layer in self.image_layers:
            x = img_layer(x)
        
        x = self.flatten_layer(x)
        non_spatial_feature = self.flatten_layer(non_spatial_feature)
        
        x = torch.cat([x, non_spatial_feature], 1)
        x = self.linear_layer1(x)
        #x = self.dropout(x)

        x = self.linear_layer2(x)
        #x = self.dropout(x)

        #x = self.linear_layer3(x)
        #x = self.dropout(x)

        x = self.sigmoid(x)

        return x