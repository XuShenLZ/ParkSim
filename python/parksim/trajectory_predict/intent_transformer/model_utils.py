from email.mime import base
import torch
from torch import default_generator, randperm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch._utils import _accumulate
from typing import Sequence, Optional, Generator, List, TypeVar
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
from parksim.intent_predict.cnnV2.pytorchtools import EarlyStopping


EARLY_STOPPING_PATH = 'models/checkpoint.pt'

_CURRENT = os.path.abspath(os.path.dirname(__file__))
T = TypeVar('T')

def random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    final = [dataset.get_subset(indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
    return final

def split_dataset(dataset, proportion_train, split_seed=42):
    train_size = int(proportion_train * len(dataset))
    val_size = len(dataset) - train_size
    validation_dataset, train_dataset = random_split(dataset, [val_size, train_size], generator=torch.Generator().manual_seed(split_seed))
    return train_dataset, validation_dataset

def generate_square_subsequent_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    #mask = mask.fill_diagonal_(float('-inf')) # Convert diagonal to -inf
    return mask

def draw_prediction(pred, inst_centric_view, X, y_label, intent):
    sensing_limit = 20
    print(f"Pred: {pred[0]}")
    print(f"Label: {y_label[0]}")
    inst_centric_view = inst_centric_view[0].permute(1,2,0).detach().cpu().numpy()
    img_size = inst_centric_view.shape[0] / 2

    traj_hist_pixel = X[0, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    traj_future_pixel = y_label[0, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    intent_pixel = intent[0, 0, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    traj_pred_pixel = pred[0, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    plt.figure()
    plt.cla()
    plt.imshow(inst_centric_view)
    plt.plot(traj_hist_pixel[:, 0], traj_hist_pixel[:, 1], 'k', linewidth=2)
    plt.plot(traj_future_pixel[:, 0], traj_future_pixel[:,
            1], 'wo', linewidth=2, markersize=2)
    plt.plot(traj_pred_pixel[:, 0], traj_pred_pixel[:, 1],
            'g^', linewidth=2, markersize=2)
    plt.plot(intent_pixel[0], intent_pixel[1], '*', color='C1', markersize=8)
    plt.axis('off')
    plt.show()


def save_model(model, path):
    config = model.get_config()
    model_info = {
        'model_state' : model.state_dict(),
        'model_config' : config,
        'model_class' : type(model)
    }
    torch.save(model_info, path)

def load_model(path, manual_class=None, manual_config=None):
    model_info = torch.load(path)
    if manual_class:
        model_class = manual_class
    else:
        model_class = model_info['model_class']

    if manual_config:
        model_config = manual_config
        model_state = model_info
    else:
        model_config = model_info['model_config']
        model_state = model_info['model_state']
    base_model = model_class(model_config)
    base_model.load_state_dict(model_state)
    return base_model


def train_loop(model, opt, loss_fn, data_loader: DataLoader, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        #X, y = get_random_batch(points.copy(), 4, 6, batch_size)
        #X, y = torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        model_input = data_loader.dataset.process_batch_training(batch, device)
        label = data_loader.dataset.process_batch_label(batch, device)
        pred = model(*model_input)
        loss = loss_fn(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.detach().item()
    return total_loss / len(data_loader)

def validation_loop(model, loss_fn, data_loader: DataLoader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            model_input = data_loader.dataset.process_batch_training(batch, device)
            label = data_loader.dataset.process_batch_label(batch, device)
            pred = model(*model_input)
            loss = loss_fn(pred, label)
            total_loss += loss.detach().item()
    return np.nanmin([total_loss / len(data_loader), 1.0e8])


def fit(model, model_name, opt, loss_fn, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs, early_stopping_patience, tensorboard, print_every, save_every, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    writer = SummaryWriter(log_dir=f'/runs/{model_name}')
    using_early_stopping = True
    if early_stopping_patience <= 0:
        using_early_stopping = False

    if using_early_stopping:
        early_stopping = EarlyStopping(patience=early_stopping_patience, path=EARLY_STOPPING_PATH, verbose=True)
    print("Training model")
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, loss_fn, train_data_loader, device)
        train_loss_list += [train_loss]
        validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
        validation_loss_list += [validation_loss]

        if tensorboard:
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
            PATH = os.path.join(_CURRENT, f'models/{model_name}_epoch_{epoch+1}_{timestamp}.pth')
            save_model(model, PATH)
        if using_early_stopping:
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_loss_list, validation_loss_list


def train_model(model, model_name, trainloader, testloader, opt, loss_fn, epochs, device, tensorboard=True, early_stopping_patience=-1, print_every=10, save_every=10):
    """
    Early stopping patience = -1 corresponds to no early stopping.
    """
    model = model.to(device)
    fit(model=model, opt=opt, loss_fn=loss_fn, train_data_loader=trainloader, val_data_loader=testloader, epochs=epochs, model_name=model_name, print_every=print_every, save_every=save_every, device=device, early_stopping_patience=early_stopping_patience, tensorboard=tensorboard)
    print('Finished Training')
    if not os.path.exists(os.path.join(_CURRENT, 'models')):
        os.mkdir(os.path.join(_CURRENT, 'models'))
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = os.path.join(_CURRENT, f'models/{model_name}_{timestamp}.pth')
    if early_stopping_patience > 0:
        model.load_state_dict(torch.load(EARLY_STOPPING_PATH))
    save_model(model, PATH)



