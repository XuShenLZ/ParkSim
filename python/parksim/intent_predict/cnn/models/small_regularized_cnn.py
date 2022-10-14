import torch
import torch.nn as nn

class SmallRegularizedCNN(nn.Module):
    """
    Simple CNN.
    """
    def __init__(self, dropout_p = 0.2):
        """
        Instantiate the model
        """
        super(SmallRegularizedCNN, self).__init__()
        
        self.image_layers = []

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(num_features=20)
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(num_features=15)
        ))
        
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3),
            nn.BatchNorm2d(num_features=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(num_features=15)s
        ))

        self.image_layer = nn.Sequential(*self.image_layers)

        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
        )
        
        IMG_LAYER_OUTPUT_SIZE = 6627
        NON_SPATIAL_FEATURE_SIZE = 5
        
        
        self.linear_layer1 = nn.Sequential(
            nn.Linear(IMG_LAYER_OUTPUT_SIZE + NON_SPATIAL_FEATURE_SIZE, 100),
            nn.BatchNorm1d(num_features=100),
            nn.Dropout(dropout_p)
            #nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(100, 1),
        )
        #self.sigmoid = nn.Sigmoid()

    def forward(self, img_feature, non_spatial_feature):
        """
        forward method
        """
        x = self.image_layer(img_feature)
        x = self.flatten_layer(x)
        non_spatial_feature = self.flatten_layer(non_spatial_feature)
        
        x = torch.cat([x, non_spatial_feature], 1)
        x = self.linear_layer1(x)
        #x = self.dropout(x)

        x = self.linear_layer2(x)
        #x = self.dropout(x)

        #x = self.linear_layer3(x)
        #x = self.dropout(x)

        #x = self.sigmoid(x)

        return x