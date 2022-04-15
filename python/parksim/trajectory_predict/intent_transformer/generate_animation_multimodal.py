from pathlib import Path
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dlp.dataset import Dataset

from parksim.intent_predict.cnnV2.data_processing.utils import CNNDataProcessor
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor

from parksim.intent_predict.cnnV2.network import SmallRegularizedCNN
from parksim.trajectory_predict.intent_transformer.network import TrajectoryPredictorWithIntent
from parksim.trajectory_predict.intent_transformer.train import build_trajectory_predict_from_config

from parksim.trajectory_predict.intent_transformer.multimodal_prediction import predict_multimodal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% 

def draw_prediction(multimodal_prediction, inst_centric_view, colors):
    sensing_limit = 20
    img_size = inst_centric_view.size[0] / 2

    plt.cla()

    plt.imshow(inst_centric_view)

    y_label, _, _, _ = multimodal_prediction[0]

    traj_future_pixel = y_label[0, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    plt.plot(traj_future_pixel[:, 0], traj_future_pixel[:,
             1], 'wo', linewidth=2, markersize=2)

    for prediction, color in zip(reversed(multimodal_prediction), reversed(colors)):

        _, pred, intent, probability = prediction

        intent_pixel = intent[0, 0, :2].detach().cpu().numpy() / \
            sensing_limit*img_size + img_size

        traj_pred_pixel = pred[0, :, :2].detach().cpu().numpy() / \
            sensing_limit*img_size + img_size

        offset = [0, 0]
        if 100 < intent_pixel[0] < 200 or 300 <= intent_pixel[0] <= 400:
            offset[0] = -40
        else:
            offset[0] = 10

        if 100 < intent_pixel[1] < 200 or 300 <= intent_pixel[1] <= 400:
            offset[1] = -20
        else:
            offset[1] = 10

        plt.plot(traj_pred_pixel[:, 0], traj_pred_pixel[:, 1],
                 '^', color=color, linewidth=2, markersize=2)
        plt.plot(intent_pixel[0], intent_pixel[1],
                 '*', color=color, markersize=8)

        plt.text(intent_pixel[0]+offset[0], intent_pixel[1]+offset[1],
                 f'{probability:.2f}', backgroundcolor=(170/255., 170/255., 170/255., 0.53), color='black', size=7, weight='bold')

    plt.axis('off')


# %%
# Load Dataset
print('loading dataset')
ds = Dataset()

home_path = str(Path.home())
ds.load(home_path + '/dlp-dataset/data/DJI_0012')

# %%
MODEL_PATH = 'models/Trajectory-Intent-4-10-22.pth'
# MODEL_PATH = 'models/checkpoint.pt'
config = {
    'dim_model': 52,
    'num_heads': 4,
    'dropout': 0.15,
    'num_encoder_layers': 16,
    'num_decoder_layers': 8,
    'd_hidden': 256,
    'num_conv_layers': 2,
    'opt': 'SGD',
    'lr': 0.0025,
    'loss': 'L1'
}

traj_model = build_trajectory_predict_from_config(config=config)
model_state = torch.load(MODEL_PATH, map_location=DEVICE)
traj_model.load_state_dict(model_state)
traj_model.eval().to(DEVICE)

INTENT_MODEL_PATH = 'models/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'
intent_model = SmallRegularizedCNN()
model_state = torch.load(INTENT_MODEL_PATH, map_location=DEVICE)
intent_model.load_state_dict(model_state)
intent_model.eval().to(DEVICE)

# %%
intent_extractor = CNNDataProcessor(ds=ds)
traj_extractor = TransformerDataProcessor(ds=ds)


# %% 

scene = ds.get('scene', ds.list_scenes()[0])
frame_index = 80
frame = ds.get_future_frames(scene['first_frame'], timesteps=300)[frame_index]
inst_token = frame['instances'][1]

agent_token = ds.get('instance', inst_token)['agent_token']
agent = ds.get('agent', agent_token)

inst_token = agent['first_instance']
inst_token_list = []
while inst_token:
    if ds.get_inst_mode(inst_token=inst_token) != 'parked':
        inst_token_list.append(inst_token)

    inst_token = ds.get('instance', inst_token)['next']

inst_token_list = inst_token_list[::10]

print(f'total number of frames = {len(inst_token_list)}')

def predict_and_draw_frame(idx):
    print(f'predicting and drawing frame {idx}')
    inst_token = inst_token_list[idx]

    multimodal_prediction, inst_centric_view = predict_multimodal(
        ds, traj_model, intent_model, traj_extractor, intent_extractor, inst_token, frame_index, 3)

    colors = ['darkviolet', 'C1', 'green']

    draw_prediction(multimodal_prediction, inst_centric_view,
                    colors)
    

fig = plt.figure()

anim = animation.FuncAnimation(fig, predict_and_draw_frame, frames=len(inst_token_list),
                               interval=0.1)

video_writer = animation.FFMpegWriter(fps=10)
anim.save('./animations/multimodal.mp4', writer=video_writer)
