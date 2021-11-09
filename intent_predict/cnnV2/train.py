from network import SimpleCNN
from utils import CNNDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import os
from datetime import datetime
import numpy as np
from network import SimpleCNN


_CURRENT = os.path.abspath(os.path.dirname(__file__))

def train_network():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    
    
    dataset = CNNDataset("data/DJI_0012", input_transform = transforms.ToTensor())
    trainloader = DataLoader(dataset, batch_size=512, shuffle=True)

    cnn = SimpleCNN().cuda()
    optimizer = optim.AdamW(cnn.parameters(), lr=1e-4, momentum=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()

    for epoch in range(100):
        running_loss = 0.0
        running_accuracy = 0.0
        for data in trainloader:
            img_feature, non_spatial_feature, labels = data
            img_feature = img_feature.cuda()
            non_spatial_feature = non_spatial_feature.cuda()
            labels = labels.cuda()
            cnn.forward(img_feature, non_spatial_feature)
            #inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            preds = cnn(img_feature, non_spatial_feature)
            labels = labels.unsqueeze(1)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(trainloader)
            predictions = (preds > 0.5).float()
            correct = (predictions == labels).float().sum() / labels.shape[0]
            running_accuracy += correct / len(trainloader)

        scheduler.step()

        # print statistics
        
        print('[%d] loss: %.3f' % (epoch + 1, running_loss ))
        print('[%d] accuracy: %.3f' % (epoch + 1, running_accuracy ))

    print('Finished Training')
    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    PATH = _CURRENT + '/models/simpleCNN_L%.3f_%s.pth' % (running_loss, timestamp)
    torch.save(cnn.state_dict(), PATH)
    

if __name__ == '__main__':
    train_network()
    
    