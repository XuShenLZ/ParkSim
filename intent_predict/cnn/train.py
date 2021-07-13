import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import os
from datetime import datetime

from utils import ImageDataset
from network import SimpleCNN
from losses import FullyConvLoss

_CURRENT = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

input_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()

batch_size = 10

trainset = ImageDataset(_CURRENT + '/data/DJI_0012', transform=input_transform, target_transform=target_transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = ImageDataset(_CURRENT + '/data/DJI_0013', transform=input_transform, target_transform=target_transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = SimpleCNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=1e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.1)

loss_fn = FullyConvLoss(lam=1)

for epoch in range(100):
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() / len(trainloader)

    scheduler.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, running_loss ))

print('Finished Training')

if not os.path.exists(_CURRENT + '/models'):
    os.mkdir(_CURRENT + '/models')

timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

PATH = _CURRENT + '/models/simpleCNN_L%.3f_%s.pth' % (running_loss, timestamp)
torch.save(model.state_dict(), PATH)