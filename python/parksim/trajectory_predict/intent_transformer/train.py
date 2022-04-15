import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
import numpy as np

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sympy import divisors
from torch.utils.tensorboard import SummaryWriter

from parksim.trajectory_predict.intent_transformer.model_utils import train_model, split_dataset, load_model
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataset
from parksim.trajectory_predict.intent_transformer.network import TrajectoryPredictorWithIntent
from parksim.intent_predict.cnnV2.pytorchtools import EarlyStopping

_CURRENT = os.path.abspath(os.path.dirname(__file__))

RUN_LABEL = 'v2'
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    config={
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.15,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
            'opt' : 'SGD',
            'lr' : 5e-4,
            'loss' : 'L1'
    }

    model = TrajectoryPredictorWithIntent(config)
    #model = load_model(model_path, manual_class=TrajectoryPredictorWithIntent)

    dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(8, 9)]
    dataset = IntentTransformerDataset(dataset_nums, img_transform=transforms.ToTensor())
    train_data, val_data = split_dataset(dataset, 0.90)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(val_data, batch_size=32, shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    loss_fn = nn.L1Loss()

    epochs = 1
    print_every=10
    save_every=50
    patience = 50

    train_model(model, "Intent-Transformer-V1", trainloader, testloader, opt, loss_fn, epochs, device, tensorboard=True, early_stopping_patience=patience, print_every=print_every, save_every=save_every)