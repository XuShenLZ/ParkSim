import torch
from vanilla_transformer.network import  TrajectoryPredictTransformerV1
from utils import CNNTransformerDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

import os
from datetime import datetime
import numpy as np
from torch import nn

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

def main():
    MODEL_PATH = "models/CNN_Transformer_03-04-2022_14-10-02.pth"
    DJI_NUM = "0008"
    DEVICE = "cuda"


    model = TrajectoryPredictTransformerV1()
    model_state = torch.load(MODEL_PATH)
    model.load_state_dict(model_state)
    model.eval().to(DEVICE)
    dataset = CNNTransformerDataset(f"data/DJI_{DJI_NUM}", img_transform = transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12)
    loss_fn = nn.MSELoss()    
    validation_loss = validation_loop(model, loss_fn, dataloader, DEVICE)
    print(f"Average Validation Loss Across Batches:\n{validation_loss}")

if __name__ == '__main__':
    main()