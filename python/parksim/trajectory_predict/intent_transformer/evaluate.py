import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from parksim.trajectory_predict.intent_transformer.network import  TrajectoryPredictorWithIntentV2
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset

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
            pred = model(img, X, intent, y_in, tgt_mask)
            loss = loss_fn(pred, y_label)
            total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def main():
    MODEL_PATH = "models\checkpoint.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    config={
            'dim_model' : 64 ,
            'num_heads' : 8 ,
            'dropout' : 0.0 ,
            'num_encoder_layers' : 4 ,
            'num_decoder_layers' : 4 ,
            'd_hidden' : 256 ,
            'num_conv_layers' : 2 ,
            'opt' : 'SGD',
            'lr' : 1e-4 ,
            'loss' : 'L1'
    }
    dataset_nums = ['../data/DJI_0012']
    model = TrajectoryPredictorWithIntentV2(config)
    model_state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(model_state)
    model.eval().to(DEVICE)
    dataset = IntentTransformerV2Dataset(dataset_nums, img_transform = transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12)
    loss_fn = nn.L1Loss()    
    validation_loss = validation_loop(model, loss_fn, dataloader, DEVICE)
    print(f"Average Validation Loss Across Batches:\n{validation_loss}")

if __name__ == '__main__':
    main()