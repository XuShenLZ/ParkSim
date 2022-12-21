import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parksim.spot_nn.feature_generator import SpotFeatureGenerator

class SpotNet(nn.Module):
    def __init__(self):
        super(SpotNet, self).__init__()
        self.feature_generator = SpotFeatureGenerator()
        self.fc1 = nn.Linear(self.feature_generator.number_of_features, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def update(self, x, y):
        self.optimizer.zero_grad()
        output = self(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
