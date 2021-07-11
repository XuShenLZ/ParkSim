import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from utils import ImageDataset
from network import IntentNet
from losses import focal_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

input_transform = transforms.ToTensor()

target_transform = transforms.ToTensor()

trainset = ImageDataset('./../training_data/DJI_0012', transform=input_transform, target_transform=target_transform)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

testset = ImageDataset('./../training_data/DJI_0013', transform=input_transform, target_transform=target_transform)
testloader = DataLoader(testset, batch_size=10, shuffle=False)

model = IntentNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.01)
l1_loss = nn.L1Loss()

for epoch in range(100):
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1]

        optimizer.zero_grad()

        preds = model(inputs)

        loss = focal_loss(preds[0].to(device), labels[0].to(device)) + l1_loss(preds[1].to(device), labels[1].to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / 10 / len(trainloader)))

print('Finished Training')

PATH = './intent_net.pth'
torch.save(model.state_dict(), PATH)