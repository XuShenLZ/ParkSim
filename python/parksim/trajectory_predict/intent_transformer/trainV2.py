import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from parksim.trajectory_predict.intent_transformer.model_utils import train_model, split_dataset
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent_v3 import TrajectoryPredictorWithIntentV3

_CURRENT = os.path.abspath(os.path.dirname(__file__))

RUN_LABEL = 'v3'
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    config={
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.10,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'patch_size' : 20,
            'lr' : 1e-4,
            'loss' : 'L1'
    }

    model = TrajectoryPredictorWithIntentV3(config)
    #model = load_model('models\Intent-Transformer-V2_epoch_299_04-13-2022_23-49-58.pth', manual_class=TrajectoryPredictorWithIntentV2)

    dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(7, 17)]
    dataset = IntentTransformerV2Dataset(dataset_nums, img_transform=transforms.ToTensor())
    train_data, val_data = split_dataset(dataset, 0.9)
    trainloader = DataLoader(train_data, batch_size=256, num_workers=12, pin_memory=True, shuffle=True)
    testloader = DataLoader(val_data, batch_size=512, num_workers=12, pin_memory=True, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], )
    loss_fn = nn.L1Loss()

    epochs = 1000
    print_every=10
    save_every=20
    patience = 50

    train_model(model, "Intent-Transformer-V3", trainloader, testloader, opt, loss_fn, epochs, device, tensorboard=True, early_stopping_patience=patience, print_every=print_every, save_every=save_every)

 