import torch
from typing import TypeVar
import matplotlib.pyplot as plt
import os
import einops
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from parksim.intent_predict.cnn.pytorchtools import EarlyStopping
from torch.utils.data import DataLoader


EARLY_STOPPING_PATH = 'models/checkpoint.pt'

_CURRENT = os.path.abspath(os.path.dirname(__file__))
T = TypeVar('T')



def split_dataset(dataset, proportions, split_seed=42):
    train_size, val_size = map(int, [prop * len(dataset) for prop in proportions[:2]])
    test_size = len(dataset) - train_size - val_size
    torch.manual_seed(split_seed)
    train_dataset, validation_dataset, testing_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(split_seed))
    return train_dataset, validation_dataset, testing_dataset

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

def tune_learning_rate(model_generator, optimizer_generator, loss_fn, learning_rates, dataloader, device, num_epochs=3, seed=42):
    best_lr_score = float('inf')
    best_lr = None
    np.random.seed(seed)
    torch.manual_seed(seed)
    t = tqdm(learning_rates)
    for lr in learning_rates:
        print(f"Current LR: {lr}")
        t.set_description(f"Learning Rate {lr}, Current Train Epoch: {-1}, Current Train Loss: {'inf'}", refresh=True)
        per_epoch_fn = lambda epoch: lambda loss: t.set_description(f"Learning Rate {lr}, Current Train Epoch: {epoch}, Current Train Loss: {loss}", refresh=True)
        model = model_generator()
        optimizer = optimizer_generator(model, lr)
        train_losses, _ = fit(model, "temp", optimizer, loss_fn, dataloader, None, num_epochs, -1, False, "/runs/", 1000, 1000, device, per_epoch_fn)
        final_train_loss = train_losses[-1]
        if final_train_loss < best_lr_score:
            best_lr_score = final_train_loss
            best_lr = lr
    print(f"BEST LR: {best_lr}")
    return best_lr

def cross_validation(model_type, configs_to_test, model_name, loss_fn, optimizer_generator, dataset, device, k_fold=5, num_epochs=5, seed=42):

    print(f"Starting {k_fold}-Fold Cross Validation With {num_epochs}-Epochs Per Model\n")
    all_cv_scores = {}
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    tensorboard_dir = f"/runs/cross_val_{timestamp}/"
    t = tqdm(enumerate(configs_to_test))
    loss_fn = loss_fn.to(device)
    for i, config in t:
        t.set_description(f"Cross Validating Model Config {i}", refresh=True)
        model_generator = lambda: model_type(config).to(device)
        train_scores, val_scores = get_cv_scores(model_generator, f"{model_name}-{i}", loss_fn, optimizer_generator, dataset, device, k_fold, num_epochs, seed, tensorboard_dir)
        all_cv_scores[i] = {
            "config" : config,
            "train_scores" : train_scores,
            "val_scores" : val_scores,
            "avg_val_score" : val_scores.mean()
        }
        print(f"Train Scores: \n{train_scores}\nValidation Scores: \n{val_scores}\nAvg Validation Score: \n{val_scores.mean()}\n======================")
    torch.save(all_cv_scores, f"data_frames/cv_results_{timestamp}.dict")
    max_cv_index = min(all_cv_scores.keys(), key=lambda i: all_cv_scores[i]["avg_val_score"])
    best_config = all_cv_scores[max_cv_index]["config"]
    best_score = all_cv_scores[max_cv_index]["avg_val_score"]
    print(f"Best Config: \n{best_config}\nBest Val Loss: \n{best_score}")
    return all_cv_scores




def get_cv_scores(model_generator, model_name, loss_fn, optimizer_generator, dataset, device, k_fold=5, num_epochs=5, seed=42, tensorboard_dir="/runs/"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 256

    train_score = pd.Series()
    val_score = pd.Series()
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    t = tqdm(range(k_fold))
    
    for i in t:
        t.set_description(f"Training Fold {i + 1}, Current Train Epoch: 0, Current Train Loss:", refresh=True)
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = dataset.get_subset(train_indices)
        val_set = dataset.get_subset(val_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=16, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=True, num_workers=16, pin_memory=True)
        model = model_generator()
        optimizer = optimizer_generator(model)
        per_epoch_fn = lambda epoch: lambda loss: t.set_description(f"Training Fold { + 1}, Current Train Epoch: {epoch}, Current Train Loss: {loss}", refresh=True)
        train_accuracies, _ = fit(model, model_name, optimizer, loss_fn, train_loader, None, num_epochs, -1, True, tensorboard_dir, 1000, 1000, device, per_epoch_fn)
        train_score.at[i] = train_accuracies[-1]
        val_acc = validation_loop(model, loss_fn, val_loader, device)
        val_score.at[i] = val_acc
    
    return train_score, val_score

def get_best_val_score(cv_result):
    max_cv_index = min(cv_result.keys(), key=lambda i: cv_result[i]["avg_val_score"])
    best_score = cv_result[max_cv_index]["avg_val_score"]
    return best_score

def get_cv_results(result_path):
    results = torch.load(result_path)
    max_cv_index = min(results.keys(), key=lambda i: results[i]["avg_val_score"])
    best_config = results[max_cv_index]["config"]
    best_score = results[max_cv_index]["avg_val_score"]
    print(f"Best Config: \n{best_config}\nBest Val Loss: \n{best_score}")
    return results

def patchify(images, patch_size=4):
    """Splitting images into patches.
    Args:
        images: Input tensor with size (batch, channels, height, width)
            We can assume that image is square where height == width.
    Returns:
        A batch of image patches with size (
          batch, (height / patch_size) * (width / patch_size), 
        channels * patch_size * patch_size)
    """
    # BEGIN YOUR CODE
    return einops.rearrange(images, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size)
    # END YOUR CODE

def unpatchify(patches, patch_size=4):
    """Combining patches into images.
    Args:
        patches: Input tensor with size (
        batch, (height / patch_size) * (width / patch_size), 
        channels * patch_size * patch_size)
    Returns:
        A batch of images with size (batch, channels, height, width)
    """
    # BEGIN YOUR CODE
    return einops.rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=int(np.sqrt(patches.shape[1])))
    # END YOUR CODE



def train_loop(model, opt, loss_fn, data_loader: DataLoader, device, per_batch_fn=lambda loss: loss):
    model.train()
    losses = []
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
        losses.append(loss.detach().item())
        current_avg = sum(losses) / len(losses)
        per_batch_fn(current_avg)
    return sum(losses) / len(losses)

def validation_loop(model, loss_fn, data_loader: DataLoader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            model_input = data_loader.dataset.process_batch_training(batch, device)
            label = data_loader.dataset.process_batch_label(batch, device)
            pred = model(*model_input)
            loss = loss_fn(pred, label)
            losses.append(loss.detach().item())
    return sum(losses) / len(losses)


def fit(model, model_name, opt, loss_fn, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs, early_stopping_patience, tensorboard, tensorboard_dir, print_every, save_every, device, per_epoch_fn=lambda epoch: lambda loss: loss):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, model_name))
    using_early_stopping = True
    if early_stopping_patience <= 0:
        using_early_stopping = False

    if using_early_stopping:
        early_stopping = EarlyStopping(patience=early_stopping_patience, path=EARLY_STOPPING_PATH, verbose=True)

    perform_validation = val_data_loader is not None and len(val_data_loader) > 0

    if not perform_validation and using_early_stopping:
        raise RuntimeError("Cannot provide empty validation loader and request early stopping")

    #print("Training model\n")
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, loss_fn, train_data_loader, device, per_batch_fn=per_epoch_fn(epoch + 1))
        train_loss_list += [train_loss]
        if perform_validation:
            validation_loss = validation_loop(model, loss_fn, val_data_loader, device)
            validation_loss_list += [validation_loss]

        if tensorboard:
            writer.add_scalar("Train Loss", train_loss, epoch)
            if perform_validation:
                writer.add_scalar("Validation Loss", validation_loss, epoch)

        if epoch % print_every == print_every - 1:
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            print(f"Training loss: {train_loss:.4f}")
            if perform_validation:
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


def train_model(model, model_name, trainloader, testloader, opt, loss_fn, epochs, device, tensorboard=True, tensorboard_dir="/runs/", early_stopping_patience=-1, print_every=10, save_every=10):
    """
    Early stopping patience = -1 corresponds to no early stopping.
    """
    model = model.to(device)
    fit(model=model, opt=opt, loss_fn=loss_fn, train_data_loader=trainloader, val_data_loader=testloader, epochs=epochs, model_name=model_name, print_every=print_every, save_every=save_every, device=device, early_stopping_patience=early_stopping_patience, tensorboard=tensorboard, tensorboard_dir=tensorboard_dir)
    print('Finished Training')
    if not os.path.exists(os.path.join(_CURRENT, 'models')):
        os.mkdir(os.path.join(_CURRENT, 'models'))
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = os.path.join(_CURRENT, f'models/{model_name}_{timestamp}.pth')
    if early_stopping_patience > 0:
        model.load_state_dict(torch.load(EARLY_STOPPING_PATH))
    save_model(model, PATH)



