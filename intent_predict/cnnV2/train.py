from network import SimpleCNN
from utils import CNNDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

import os
from datetime import datetime
import numpy as np
from network import SimpleCNN
from sklearn.model_selection import KFold
from pytorchtools import EarlyStopping


_CURRENT = os.path.abspath(os.path.dirname(__file__))

def train_network():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    
    
    train_dataset = CNNDataset("data/DJI_0012", input_transform = transforms.ToTensor())
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    full_validation_dataset = CNNDataset("data/DJI_0005", input_transform = transforms.ToTensor())
    val_size = int(0.3 * len(full_validation_dataset))
    unused_data_size = len(full_validation_dataset) - val_size
    
    validation_dataset, _ = torch.utils.data.random_split(full_validation_dataset, [val_size, unused_data_size], generator=torch.Generator().manual_seed(42))
    testloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=12)

    cnn = SimpleCNN()
    cnn = cnn.cuda()
    optimizer = optim.AdamW(cnn.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
    early_stopping = EarlyStopping(patience=10, path= 'models/checkpoint.pt', verbose=True)
    
    num_epochs = 50
    

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_train_accuracy = 0.0
        for data in tqdm(trainloader):
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
            running_train_accuracy += correct / len(trainloader)
        
        running_val_accuracy = 0
        with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    img_feature, non_spatial_feature, labels = data
                    img_feature = img_feature.cuda()
                    non_spatial_feature = non_spatial_feature.cuda()
                    labels = labels.cuda().unsqueeze(1)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Perform forward pass
                    preds = cnn(img_feature, non_spatial_feature)
                    
                    # Compute loss
                    loss = loss_fn(preds, labels)

                    # Set total and correct
                    predictions = (preds > 0.5).float()
                    correct = (predictions == labels).float().sum() / labels.shape[0]
                    running_val_accuracy += correct / len(testloader)
        
        early_stopping(running_val_accuracy, cnn)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # print statistics
        
        print('[%d] loss: %.3f' % (epoch + 1, running_loss ))
        print('[%d] train accuracy: %.3f' % (epoch + 1, running_train_accuracy ))
        print('[%d] validation accuracy: %.3f' % (epoch + 1, running_val_accuracy ))

    print('Finished Training')
    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = _CURRENT + '/models/simpleCNN_L%.3f_%s.pth' % (running_loss, timestamp)
    cnn.load_state_dict(torch.load(early_stopping.path))
    torch.save(cnn.state_dict(), PATH)
    

if __name__ == '__main__':
    train_network()
    
    