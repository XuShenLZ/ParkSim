from time import time
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import os
from datetime import datetime
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sympy import divisors
import random

from parksim.trajectory_predict.vanilla_transformer.dataset import CNNTransformerDatasetMulti
from parksim.trajectory_predict.vanilla_transformer.network import TrajectoryPredictTransformerV1

_CURRENT = os.path.abspath(os.path.dirname(__file__))

def train_loop(model, opt, loss_fn, data_loader, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        #X, y = get_random_batch(points.copy(), 4, 6, batch_size)
        #X, y = torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        img, X, y_in, y_label = batch
        img = img.to(device).float()
        X = X.to(device).float()
        y_in = y_in.to(device).float()
        y_label = y_label.to(device).float()
        tgt_mask = model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(device).float()

        # Standard training except we pass in y_input and tgt_mask
        pred = model(img, X, y_in, tgt_mask=tgt_mask)
        # Permute pred to have batch size first again
        loss = loss_fn(pred, y_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.detach().item()
        
    return total_loss / len(data_loader)

def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            img, X, y_in, y_label = batch
            img = img.to(device).float()
            X = X.to(device).float()
            y_in = y_in.to(device).float()
            y_label = y_label.to(device).float()
            tgt_mask = model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(device).float()
            pred = model(img, X, y_in, tgt_mask)
            loss = loss_fn(pred, y_label)
            total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def write_model_graph(model, data_loader, writer):
    with torch.no_grad():
        for batch in data_loader:
            writer.add_graph(model, input_to_model=batch[0], verbose=False)
            break

def fit(model, opt, loss_fn, train_data_loader, val_data_loader, epochs, model_name, print_every=10, save_every=100, device="cuda"):
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    print("Training model")
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, loss_fn, train_data_loader, device)
        train_loss_list += [train_loss]
        validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
        validation_loss_list += [validation_loss]

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), opt.state_dict()), path)
        tune.report(train_loss=train_loss, validation_loss=validation_loss)


        if epoch % print_every == print_every - 1:
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
        if epoch % save_every == save_every - 1:
            if not os.path.exists(_CURRENT + '/models'):
                os.mkdir(_CURRENT + '/models')
            timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            PATH = _CURRENT + f'../models/{model_name}_epoch_{epoch}_{timestamp}.pth'
            torch.save(model.state_dict(), PATH)

    return train_loss_list, validation_loss_list

def build_trajectory_predict_from_config(config, input_shape=(3, 100, 100)):
    model = TrajectoryPredictTransformerV1(
        input_shape=input_shape,
        dropout=config['dropout'], 
        num_heads=config['num_heads'], 
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers'], 
        dim_model=config['dim_model'],
        num_conv_layers=config['num_conv_layers']
    )
    return model

def train_model(config, dataset_nums, epochs, save_every, device, model_name, writer=None):

    val_proportion = 0.25
    seed = 42

    model = build_trajectory_predict_from_config(config)
    model = model.to(device)

    dataset = CNNTransformerDatasetMulti(dataset_nums, img_transform=transforms.ToTensor())
    val_proportion = 0.20
    val_size = int(val_proportion * len(dataset))
    train_size = len(dataset) - val_size
    validation_dataset, train_dataset = torch.utils.data.random_split(dataset, [val_size, train_size], generator=torch.Generator().manual_seed(seed))
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    testloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=12)

    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=config['lr'])
    if writer:
        write_model_graph(model, trainloader, writer)
    fit(model=model, opt=opt, loss_fn=loss_fn, train_data_loader=trainloader, val_data_loader=testloader, epochs=epochs, model_name=model_name, print_every=10, save_every=save_every, device=device)

    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = _CURRENT + f'/models/{model_name}_{timestamp}.pth'
    torch.save(model.state_dict(), PATH)



RUN_LABEL = 'v1'


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print()
    #writer = SummaryWriter(logdir=f'/runs/{RUN_LABEL}')

    # dataset_nums = ['data/DJI_0008', 'data/DJI_0009', 'data/DJI_0010', 'data/DJI_0011', 'data/DJI_0012']
    dataset_nums = ['data/DJI_0012']
    #print(os.path.exists(dataset_nums[0]))
    #exit()
    epochs = 50
    NUM_SAMPLES = 100
    #model_state = torch.load('models/CNN_Transformer_03-04-2022_13-58-49.pth')
    #model = TrajectoryPredictTransformerV1().to(device)
    #model.load_state_dict(model_state)

    #HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300, 200, 512]))
    
    config={
        'dim_model' : tune.qrandint(16, 64, q=4),
        'num_heads' : tune.sample_from(lambda spec: random.choice(divisors(spec.config.dim_model))),
        'dropout' : tune.uniform(0.1, 0.5),
        'num_encoder_layers' : tune.qrandint(4, 16, q=2),
        'num_decoder_layers' : tune.qrandint(4, 16, q=2),
        'num_conv_layers' : tune.randint(0, 9),
        'lr' : tune.loguniform(1e-2, 1e-1)
    }


    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        time_attr="training_iteration",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["train_loss", "validation_loss", "training_iteration"])
    result = tune.run(
        partial(train_model, dataset_nums=dataset_nums, epochs=epochs, save_every=50, device=device, model_name='CNN_Transformer', writer=None),
        config=config,
        resources_per_trial={"cpu": 32, "gpu": 2},
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("validation_loss", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial training loss: {}".format(
        best_trial.last_result["train_loss"]))
    print("Best trial validation loss: {}".format(
        best_trial.last_result["validation_loss"]))
    
    best_trained_model = build_trajectory_predict_from_config(best_trial.config)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    print("Best Checkpoint Dir: {}".format(best_checkpoint_dir))

