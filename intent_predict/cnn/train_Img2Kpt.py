import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import os
from datetime import datetime

from utils import ImgDataset
from network import KeypointNet

_CURRENT = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

input_transform = transforms.ToTensor()

def kpt_normalize(keypoint):
    return torch.from_numpy(keypoint / 400).float()

batch_size = 10

trainset = ImgDataset(_CURRENT + '/data/DJI_0012', transform=input_transform, target_transform=kpt_normalize)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

valset = ImgDataset(_CURRENT + '/data/DJI_0013', transform=input_transform, target_transform=kpt_normalize)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

model = KeypointNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.5)

loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    train_loss = 0.0
    model.train()
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)

    valid_loss = 0.0
    model.eval()
    for data in valloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        preds = model(inputs)

        loss = loss_fn(preds, labels)

        valid_loss += loss.item() / len(valloader)
    
    scheduler.step()

    # print statistics
    print('[%d] training loss: %.3f, valid loss: %.3f' % (epoch + 1, train_loss, valid_loss ))

print('Finished Training')

if not os.path.exists(_CURRENT + '/models'):
    os.mkdir(_CURRENT + '/models')

timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

PATH = _CURRENT + '/models/KeypointNet_L%.3f_%s.pth' % (valid_loss, timestamp)
torch.save(model.state_dict(), PATH)