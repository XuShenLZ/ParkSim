from time import time
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import os
from datetime import datetime

from parksim.trajectory_predict.utils import CNNTransformerDatasetMulti
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

def fit(model, opt, loss_fn, train_data_loader, val_data_loader, epochs, print_every=10, save_every=100, device="cuda"):
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    print("Training model")
    for epoch in range(epochs):
        if epoch % print_every == print_every - 1:
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            train_loss = train_loop(model, opt, loss_fn, train_data_loader, device)
            train_loss_list += [train_loss]
            validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
            validation_loss_list += [validation_loss]
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
        else:
            train_loss = train_loop(model, opt, loss_fn, train_data_loader, device)
            train_loss_list += [train_loss]
            validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
            validation_loss_list += [validation_loss]
        if epoch % save_every == save_every - 1:
            if not os.path.exists(_CURRENT + '/models'):
                os.mkdir(_CURRENT + '/models')
            timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            PATH = _CURRENT + f'/models/CNN_Transformer_{timestamp}.pth'
            torch.save(model.state_dict(), PATH)

    return train_loss_list, validation_loss_list

def train_model(model, dataset_nums, epochs, save_every, device):

    val_proportion = 0.25
    seed = 42
    model = model.to(device)

    dataset = CNNTransformerDatasetMulti(dataset_nums, img_transform=transforms.ToTensor())
    val_proportion = 0.20
    val_size = int(val_proportion * len(dataset))
    train_size = len(dataset) - val_size
    validation_dataset, train_dataset = torch.utils.data.random_split(dataset, [val_size, train_size], generator=torch.Generator().manual_seed(seed))
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    testloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=12)

    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    fit(model=model, opt=opt, loss_fn=loss_fn, train_data_loader=trainloader, val_data_loader=testloader, epochs=epochs, print_every=10, save_every=save_every, device=device)
    print('Finished Training')


    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = _CURRENT + f'/models/CNN_Transformer_all_data_{timestamp}.pth'
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print()


    # dataset_nums = ['data/DJI_0008', 'data/DJI_0009', 'data/DJI_0010', 'data/DJI_0011', 'data/DJI_0012']
    dataset_nums = ['data/DJI_0012']
    epochs = 600

    #model_state = torch.load('models/CNN_Transformer_03-04-2022_13-58-49.pth')
    model = TrajectoryPredictTransformerV1().to(device)
    #model.load_state_dict(model_state)

    train_model(model=model, dataset_nums=dataset_nums, epochs=epochs, save_every=50, device=device)

