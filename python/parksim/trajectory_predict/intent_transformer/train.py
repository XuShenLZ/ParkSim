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


from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataset
from parksim.trajectory_predict.intent_transformer.network import TrajectoryPredictorWithIntent
from parksim.intent_predict.cnnV2.pytorchtools import EarlyStopping

_CURRENT = os.path.abspath(os.path.dirname(__file__))

def train_loop(model, opt, loss_fn, data_loader, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        #X, y = get_random_batch(points.copy(), 4, 6, batch_size)
        #X, y = torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        img, X, intent, y_in, y_label = batch
        img = img.to(device).float()
        X = X.to(device).float()
        intent = intent.to(device).float()
        y_in = y_in.to(device).float()
        y_label = y_label.to(device).float()
        tgt_mask = model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(device).float()

        # Standard training except we pass in y_input and tgt_mask
        pred = model(img, X, intent, y_in, tgt_mask)
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
            img, X, intent, y_in, y_label = batch
            img = img.to(device).float()
            X = X.to(device).float()
            intent = intent.to(device).float()
            y_in = y_in.to(device).float()
            y_label = y_label.to(device).float()
            tgt_mask = model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(device).float()
            pred = model(img, X, intent, y_in, tgt_mask=tgt_mask)
            loss = loss_fn(pred, y_label)
            total_loss += loss.detach().item()
    return np.nanmin([total_loss / len(dataloader), 1.0e8])

def write_model_graph(model, data_loader, writer):
    with torch.no_grad():
        for batch in data_loader:
            writer.add_graph(model, input_to_model=batch[0], verbose=False)
            break

def fit(model, opt, loss_fn, train_data_loader, val_data_loader, epochs, model_name, print_every=10, save_every=100, device="cuda", early_stopping=None, writer=None):
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    print("Training model")
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, loss_fn, train_data_loader, device)
        train_loss_list += [train_loss]
        validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
        validation_loss_list += [validation_loss]

        if writer:
            writer.add_scalar("Train Loss", train_loss, epoch)
            writer.add_scalar("Validation Loss", validation_loss, epoch)

        if epoch % print_every == print_every - 1:
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
        if epoch % save_every == save_every - 1:
            if not os.path.exists(os.path.join(_CURRENT, 'models')):
                os.mkdir(os.path.join(_CURRENT, 'models'))
            timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            PATH = os.path.join(_CURRENT, f'models/{model_name}_epoch_{epoch}_{timestamp}.pth')
            torch.save(model.state_dict(), PATH)
        if early_stopping:
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_loss_list, validation_loss_list

def build_trajectory_predict_from_config(config, input_shape=(3, 100, 100)):
    model = TrajectoryPredictorWithIntent(
        input_shape=input_shape,
        dropout=config['dropout'], 
        num_heads=config['num_heads'], 
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers'], 
        dim_model=config['dim_model'],
        d_hidden=config['d_hidden'],
        num_conv_layers=config['num_conv_layers']
    )
    return model

def train_model(config, dataset_nums, epochs, save_every, device, model_name, writer=None, early_stopping=None):

    val_proportion = 0.1
    seed = 42
    model = build_trajectory_predict_from_config(config)
    model = model.to(device)

    if 'previous_path' in config:
        model_state = torch.load(config['previous_path'], map_location=device)
        model.load_state_dict(model_state)

    if config['opt'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['opt'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    dataset = IntentTransformerDataset(dataset_nums, img_transform=transforms.ToTensor())
    val_size = int(val_proportion * len(dataset))
    train_size = len(dataset) - val_size
    validation_dataset, train_dataset = torch.utils.data.random_split(dataset, [val_size, train_size], generator=torch.Generator().manual_seed(seed))
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    testloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=1)

    if config['loss'] == "L2":
        loss_fn = nn.MSELoss()
    elif config['loss'] == "L1":
        loss_fn = nn.L1Loss()

    fit(model=model, opt=opt, loss_fn=loss_fn, train_data_loader=trainloader, val_data_loader=testloader, epochs=epochs, model_name=model_name, print_every=10, save_every=save_every, device=device, early_stopping=early_stopping, writer=writer)
    print('Finished Training')
    if not os.path.exists(os.path.join(_CURRENT, 'models')):
        os.mkdir(os.path.join(_CURRENT, 'models'))
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = os.path.join(_CURRENT, f'models/{model_name}_{timestamp}.pth')
    model.load_state_dict(torch.load(early_stopping.path))
    torch.save(model.state_dict(), PATH)





RUN_LABEL = 'v1'
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print()
    writer = SummaryWriter(log_dir=f'runs/{RUN_LABEL}')

    # dataset_nums = ['data/DJI_0008', 'data/DJI_0009', 'data/DJI_0010', 'data/DJI_0011', 'data/DJI_0012']
    #print(os.path.exists(dataset_nums[0]))
    #exit()
    #model_state = torch.load('models/CNN_Transformer_03-04-2022_13-58-49.pth')
    #model = TrajectoryPredictTransformerV1().to(device)
    #model.load_state_dict(model_state)

    #HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300, 200, 512]))
    
    config={
            'dim_model' : 52,
            'num_heads' : 4,
            'dropout' : 0.15,
            'num_encoder_layers' : 16,
            'num_decoder_layers' : 8,
            'd_hidden' : 256,
            'num_conv_layers' : 2,
            'opt' : 'SGD',
            'lr' : 0.0025,
            'loss' : 'L1'
    }

    dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(7, 22)]
    epochs = 1000
    save_every=50
    patience = 100
    early_stopping = EarlyStopping(patience=patience, path='models/checkpoint.pt', verbose=True)

    train_model(config=config, dataset_nums=dataset_nums, epochs=epochs, save_every=save_every, device=device, model_name="Intent_Transformer", early_stopping=early_stopping, writer=writer)
    writer.flush()

 