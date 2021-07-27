import torch
import torch.nn as nn

from torchvision import models

from torchvision.models import inception

class SimpleCNN(nn.Module):
    """
    Simple CNN. No deconvolution
    """
    def __init__(self):
        """
        Instantiate the model
        """
        super(SimpleCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=6)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        forward method
        """
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.maxpool(x)

        x = self.sigmoid(x)

        return x

class KeypointNet(nn.Module):
    """
    From Image to Keypoint location
    """
    def __init__(self):
        super(KeypointNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=12),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=6*6*16, out_features=120),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=30),
            nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=30, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        forward pass
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


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