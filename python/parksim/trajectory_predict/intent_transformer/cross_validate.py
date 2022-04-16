import torch
from torchvision import transforms
from torch import nn
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset
from parksim.trajectory_predict.intent_transformer.network import TrajectoryPredictorWithIntentV2
from parksim.trajectory_predict.intent_transformer.model_utils import cross_validation

configs_to_test = [
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.2,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.2,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.3,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.3,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.1,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.1,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 3,
        },
        
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.2,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.2,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.3,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.3,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.1,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
        {
            'dim_model' : 256,
            'num_heads' : 16,
            'dropout' : 0.1,
            'num_encoder_layers' : 12,
            'num_decoder_layers' : 12,
            'd_hidden' : 256,
            'num_conv_layers' : 6,
        },
    ]

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_type = TrajectoryPredictorWithIntentV2
    model_name = "IntentTransformerV2"
    lr = 1e-3
    optimizer_generator = lambda model: torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(12, 13)]
    dataset = IntentTransformerV2Dataset(dataset_nums, img_transform=transforms.ToTensor())
    cross_validation(model_type=model_type, configs_to_test=configs_to_test, model_name=model_name, loss_fn=loss_fn, optimizer_generator=optimizer_generator, dataset=dataset, device=device)