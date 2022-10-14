import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent_v3 import TrajectoryPredictorWithIntentV3
from parksim.trajectory_predict.intent_transformer.model_utils import tune_learning_rate

config = {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.1,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'patch_size' : 20,
        }

learning_rates_to_test = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(12, 13)]
    dataset = IntentTransformerV2Dataset(dataset_nums, img_transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=12)
    model_generator = lambda : TrajectoryPredictorWithIntentV3(config).to(device)
    optimizer_generator = lambda model, lr: torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss().to(device)
    best_lr = tune_learning_rate(model_generator, optimizer_generator, loss_fn, learning_rates_to_test, dataloader, device, num_epochs=5)
    print(f"BEST LR: {best_lr}")