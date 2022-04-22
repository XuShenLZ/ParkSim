import torch
from torch import Tensor
from torchvision import transforms

from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import numpy as np
from tqdm import tqdm
import multiprocessing

import os

from dlp.dataset import Dataset
from parksim.trajectory_predict.intent_transformer.network import TrajectoryPredictorWithIntent, TrajectoryPredictorWithIntentV4
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor
from parksim.trajectory_predict.intent_transformer.model_utils import generate_square_subsequent_mask


def generate_data_for_agent_in_range(extractor: TransformerDataProcessor, instances, start_idx: int, stride: int=10, history: int=10, future: int=10, img_size: int=100):
        img_transform=transforms.ToTensor()
        all_image_history = []
        all_trajectory_history = []
        all_trajectory_future = []
        all_trajectory_future_tgt = []
        all_local_intent_pose = []
        all_inst_centric_view = []
        inst_idx = start_idx
        curr_instance = instances[inst_idx]
        inst_token = curr_instance['instance_token']
        img_frame = extractor.vis.plot_frame(curr_instance['frame_token'])
        image_feature = extractor.vis.inst_centric(img_frame, inst_token)

        global_intent_pose = extractor.get_intent_pose(
            inst_token=inst_token, inst_centric_view=image_feature)

        image_feature = extractor.label_target_spot(inst_token, image_feature)

        all_inst_centric_view = image_feature.copy()

        curr_pose = np.array([curr_instance['coords'][0],
                                curr_instance['coords'][1], curr_instance['heading']])
        rot = np.array([[np.cos(-curr_pose[2]), -np.sin(-curr_pose[2])],
                    [np.sin(-curr_pose[2]), np.cos(-curr_pose[2])]])

        local_intent_coords = np.dot(rot, global_intent_pose[:2]-curr_pose[:2])
        local_intent_pose = np.array(
            [local_intent_coords[0], local_intent_coords[1]])
        local_intent_pose = np.expand_dims(local_intent_pose, axis=0)

        image_history = []
        trajectory_history = []
        for i in range(inst_idx - stride * (history - 1), inst_idx+1, stride):
            instance = instances[i]
            pos = np.array(instance['coords'])
            translated_pos = np.dot(rot, pos-curr_pose[:2])
            trajectory_history.append(Tensor(
                [translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))

            # generate image history
            img_frame = extractor.vis.plot_frame(instance['frame_token'])
            image_feature = extractor.vis.inst_centric(
                img_frame, instance['instance_token'], curr_pose)
            image_feature = extractor.label_target_spot(
                inst_token, image_feature, curr_pose)

            # Image transformation
            image_tensor = img_transform(image_feature.resize((img_size, img_size)))
            image_history.append(image_tensor)
        
        trajectory_future = []
        for i in range(inst_idx + stride, inst_idx + stride * future + 1, stride):
            instance = instances[i]
            pos = np.array(instance['coords'])
            translated_pos = np.dot(rot, pos-curr_pose[:2])
            trajectory_future.append(Tensor(
                [translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))
        
        all_image_history = torch.stack(image_history)
        all_trajectory_history = torch.stack(trajectory_history)
        # This is the tgt that is passed into the decoder, and trajectory_future is the label
        trajectory_future_tgt = torch.stack(
            trajectory_history[-1:] + trajectory_future[:-1])
        all_trajectory_future_tgt = trajectory_future_tgt
        all_trajectory_future = torch.stack(trajectory_future)
        all_local_intent_pose = torch.from_numpy(local_intent_pose)

        return all_image_history, all_trajectory_history, all_local_intent_pose, all_trajectory_future_tgt, all_trajectory_future, all_inst_centric_view

def generate_data_for_agent(agent_token: str, extractor: TransformerDataProcessor, stride: int=10, history: int=10, future: int=10, img_size: int=100):
    instances = ds.get_agent_instances(agent_token)
    start_idx = history * stride
    end_idx = len(instances) - 1 - future * stride

    num_points = (end_idx - start_idx) // stride
    all_inputs = tqdm([(extractor, instances, start_idx + i * stride, stride, history, future, img_size) for i in range(num_points)])
    with multiprocessing.Pool(4) as pool:
        result = pool.starmap(generate_data_for_agent_in_range, all_inputs)
    get_ith_index_list = lambda i : lambda t : t[i]
    map_lam = lambda fun, lst : list(map(fun, lst))
    return torch.stack(map_lam(get_ith_index_list(0), result)), torch.stack(map_lam(get_ith_index_list(1), result)), torch.stack(map_lam(get_ith_index_list(2), result)), torch.stack(map_lam(get_ith_index_list(3), result)), torch.stack(map_lam(get_ith_index_list(4), result)), map_lam(get_ith_index_list(5), result)


def draw_prediction(idx):
    sensing_limit = 20
    inst_centric_view = list_inst_centric_view[idx]
    img_size = inst_centric_view.size[0] / 2

    traj_hist_pixel = X[idx, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    traj_future_pixel = y_label[idx, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    intent_pixel = intent[idx, 0, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    traj_pred_pixel = pred[idx, :, :2].detach().cpu().numpy() / \
        sensing_limit*img_size + img_size

    plt.cla()
    plt.imshow(inst_centric_view)
    plt.plot(traj_hist_pixel[:, 0], traj_hist_pixel[:, 1], 'k', linewidth=2)
    plt.plot(traj_future_pixel[:, 0], traj_future_pixel[:,
            1], 'wo', linewidth=2, markersize=2)
    plt.plot(traj_pred_pixel[:, 0], traj_pred_pixel[:, 1],
            'g^', linewidth=2, markersize=2)
    plt.plot(intent_pixel[0], intent_pixel[1], '*', color='C1', markersize=8)
    plt.axis('off')

if __name__ == '__main__':    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    MODEL_PATH = r"C:\Users\rlaca\Documents\GitHub\ParkSim\python\parksim\trajectory_predict\intent_transformer\checkpoints\v4\lightning_logs\version_8\checkpoints\epoch=101-val_loss=0.0320.ckpt"
    model = TrajectoryPredictorWithIntentV4.load_from_checkpoint(MODEL_PATH)
    model.eval().to(DEVICE)

    dji_num = '0012'

    home_path = Path.home() / 'Documents/GitHub'
    # Load dataset
    ds = Dataset()
    ds.load(str(home_path / f'dlp-dataset/data/DJI_{dji_num}'))

    extractor = TransformerDataProcessor(ds=ds)
    scene = ds.get('scene', ds.list_scenes()[0])
    agents = scene['agents']
    for agent_idx in range(len(agents)):
        print(agent_idx)
        if agent_idx == 0:
            continue
        try:
            agent_token = agents[agent_idx]
            agent_type = ds.get('agent', agent_token)['type']
            if agent_type != "Car":
                continue
            img, X, intent, y_in, y_label, list_inst_centric_view = generate_data_for_agent(agent_token=agent_token, extractor=extractor)
            with torch.no_grad():
                img = img[:, -1].to(DEVICE).float()
                X = X.to(DEVICE).float()
                intent = intent.to(DEVICE).float()
                y_in = y_in.to(DEVICE).float()
                y_label = y_label.to(DEVICE).float()
                tgt_mask = generate_square_subsequent_mask(
                    y_in.shape[1]).to(DEVICE).float()
                pred = model(img, X, intent, y_in, tgt_mask=tgt_mask)
                del img
            
            fig = plt.figure()

            anim = animation.FuncAnimation(fig, draw_prediction, frames=pred.shape[0],
                                        interval=0.1)
            fname = f'C:\\Users\\rlaca\\Documents\\GitHub\\ParkSim\\python\\parksim\\trajectory_predict\\intent_transformer\\animations\\animation-dji-{dji_num}-agent-{agent_idx}-v2.mp4'
            video_writer = animation.FFMpegWriter(fps=10)
            anim.save(fname, writer=video_writer)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)  
            continue