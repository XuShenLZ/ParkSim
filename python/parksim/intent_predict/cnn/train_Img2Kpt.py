import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.summary import hparams
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from itertools import product
from tqdm import tqdm

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

loss_fn = torch.nn.MSELoss()

def train(hparams):

    print(hparams)


    writer = SummaryWriter()

    model = KeypointNet(dropout_p=hparams['dropout']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=hparams['lr'], momentum=hparams['momentum'], weight_decay=hparams['wdecay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[15, 25, 40], gamma=hparams['gamma'])

    for epoch in tqdm(range(50)):
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

        writer.add_scalars("Losses", {'train': train_loss, 'val': valid_loss}, epoch)

    # print statistics
    print('[%d] training loss: %.3f, valid loss: %.3f' % (epoch + 1, train_loss, valid_loss ))

    writer.add_hparams(hparam_dict=hparams, metric_dict={'train_loss': train_loss, 'val_loss': valid_loss})

    print('Finished Training')
    writer.flush()

    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    PATH = _CURRENT + '/models/KeypointNet_L%.3f_%s.pth' % (valid_loss, timestamp)
    torch.save(model.state_dict(), PATH)
    writer.close()

if __name__ == "__main__":
    # lr_list = [1e-4, 1e-3, 1e-2]
    # momentum_list = [1e-4, 1e-3, 1e-2]
    # wdecay_list = [1e-4, 1e-3, 1e-2]
    # dropout_list = [0.1, 0.2, 0.4]
    # gamma_list = [0.1, 0.2, 0.5]

    # for lr, momentum, wdecay, dropout, gamma in product(lr_list, momentum_list, wdecay_list, dropout_list, gamma_list):
    #     hparams = {'lr': lr,
    #                 'momentum': momentum,
    #                 'wdecay': wdecay,
    #                 'dropout': dropout,
    #                 'gamma': gamma}

    #     train(hparams)

    hparams = {'lr': 1e-3,
                'momentum': 1e-4,
                'wdecay': 1e-4,
                'dropout': 0.1,
                'gamma': 0.5}

    train(hparams)