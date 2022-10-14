import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from parksim.trajectory_predict.intent_transformer.model_utils import validation_loop
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent_v2 import TrajectoryPredictorWithIntentV2
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset

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