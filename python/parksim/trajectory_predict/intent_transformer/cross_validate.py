import torch
from torchvision import transforms
from torch import nn
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2Dataset
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent_v3 import TrajectoryPredictorWithIntentV3
from parksim.trajectory_predict.intent_transformer.model_utils import cross_validation, get_best_val_score

configs_to_test = [
        {
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.1,
            'num_encoder_layers' : 6,
            'num_decoder_layers' : 6,
            'd_hidden' : 256,
            'patch_size' : 20,
        },
    ]

learning_rates_to_test = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_type = TrajectoryPredictorWithIntentV3
    model_name = "IntentTransformerV3"
    best_lr_score = float('inf')
    best_lr = None
    for lr in learning_rates_to_test:
        print(f"Current LR: {lr}")
        optimizer_generator = lambda model: torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(12, 14)]
        dataset = IntentTransformerV2Dataset(dataset_nums, img_transform=transforms.ToTensor())
        result = cross_validation(model_type=model_type, configs_to_test=configs_to_test, model_name=model_name, loss_fn=loss_fn, optimizer_generator=optimizer_generator, dataset=dataset, device=device, num_epochs=3, k_fold=5
        best_score = get_best_val_score(result)
        if best_score < best_lr_score:
            best_lr_score = best_score
            best_lr = lr
    print(f"BEST LR: {best_lr}")