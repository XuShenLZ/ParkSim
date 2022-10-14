# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sns.set_theme(style="darkgrid")

# %% [markdown]
# # Transformer with Intent

# %%
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent import TrajectoryPredictorWithIntent
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataset

# %%
def build_trajectory_predict_from_config(config, input_shape=(3, 100, 100)):
    model = TrajectoryPredictorWithIntent(
        input_shape=input_shape,
        dropout=config['dropout'], 
        num_heads=config['num_heads'], 
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers'], 
        dim_model=config['dim_model'],
        d_hidden=config['d_hidden'],
        num_conv_layers=config['num_conv_layers']
    )
    return model

# %%
dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(7, 23)]
seed=42
val_proportion = 0.1
dataset = IntentTransformerDataset(dataset_nums, img_transform=transforms.ToTensor())
val_size = int(val_proportion * len(dataset))
train_size = len(dataset) - val_size
validation_dataset, _ = torch.utils.data.random_split(dataset, [val_size, train_size], generator=torch.Generator().manual_seed(seed))
dataloader = DataLoader(validation_dataset, batch_size=32, num_workers=1)
loss_fn = nn.L1Loss(reduction='none').to(DEVICE)


# %%
model_paths = {"all": {
    "path": "models/Trajectory-Intent-4-10-22.pth",
    "config" : {
            'dim_model' : 52,
            'num_heads' : 4,
            'dropout' : 0.15,
            'num_encoder_layers' : 16,
            'num_decoder_layers' : 8,
            'd_hidden' : 256,
            'num_conv_layers' : 2,
            'opt' : 'SGD',
            'lr' : 0.0025,
            'loss' : 'L2'
}
    }}


# %%
def get_error_vs_time(model, dataloader, loss_fn, feature_size=3, steps=10):
    model.eval()

    pos_error = torch.empty(size=(feature_size, steps)).to(DEVICE)
    ang_error = torch.empty(size=(feature_size, steps)).to(DEVICE)

    with torch.no_grad():
        for batch in dataloader:
            img, X, intent, y_in, y_label = batch
            img = img.to(DEVICE).float()
            X = X.to(DEVICE).float()
            intent = intent.to(DEVICE).float()
            y_in = y_in.to(DEVICE).float()
            y_label = y_label.to(DEVICE).float()
            tgt_mask = model.transformer.generate_square_subsequent_mask(
                y_in.shape[1]).to(DEVICE).float()
            pred = model(img, X, intent, y_in, tgt_mask)
            loss = loss_fn(pred, y_label)

            pos_error = torch.cat(
                [pos_error, torch.sqrt(loss[:, :, 0]**2 + loss[:, :, 1]**2).detach()])
            ang_error = torch.cat([ang_error, loss[:, :, 2].detach()])

    return pos_error.cpu().numpy(), ang_error.cpu().numpy()


# %%
all_error = []
dt = 0.4

for name, model_info in model_paths.items():
    print(f'Getting statistics for model {name}')

    model = build_trajectory_predict_from_config(model_info['config'])
    model_state = torch.load(model_info['path'], map_location=DEVICE)
    model.load_state_dict(model_state)
    model.eval().to(DEVICE)

    pos_error, ang_error = get_error_vs_time(model, dataloader, loss_fn, steps=10)

    timesteps = dt*np.arange(1, pos_error.shape[1]+1)
    for i, time in enumerate(timesteps):
        for error in pos_error[:, i]:
            all_error.append([name, time, 'Positional', error])
        for error in ang_error[:, i]:
            all_error.append([name, time, 'Angular', error])

error_df = pd.DataFrame(
    all_error, columns=['Epoch', 'Timestep', 'Type', 'Error'])
error_df = error_df[abs(error_df['Error']) < 100] # Outlier removal
error_df['Epoch'] = pd.Categorical(error_df.Epoch)
error_df['Type'] = pd.Categorical(error_df.Type)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.lineplot(x="Timestep", y="Error", hue="Epoch", ci=95, style="Epoch",
             markers=True, dashes=False, data=error_df[error_df["Type"] == "Positional"], ax=axes[0])
sns.lineplot(x="Timestep", y="Error", hue="Epoch", ci=95, style="Epoch",
             markers=True, dashes=False, data=error_df[error_df["Type"] == "Angular"], ax=axes[1])

axes[0].set_title('Positional Error (m)')
axes[1].set_title('Angular Error (rad)')
plt.savefig('error_over_time.png')
# %%



