import argparse
import multiprocessing
import numpy as np
import os

from dlp.dataset import Dataset
from itertools import product
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

stride = 25
history = 3
future = 3

_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def get_data_for_instance(inst_token: str, inst_idx: int, frame_token: str, extractor: TransformerDataProcessor, ds: Dataset) -> Tuple[np.array, np.array, np.array]:
    """
    returns image, trajectory_history, and trajectory future for given instance
    """
    img_frame = extractor.vis.plot_frame(frame_token)
    image_feature = extractor.vis.inst_centric(img_frame, inst_token)
    image_feature = extractor.label_target_spot(inst_token, image_feature)

    curr_instance = ds.get('instance', inst_token)
    current_state = np.array([curr_instance['coords'][0], curr_instance['coords'][1], curr_instance['heading'], curr_instance['speed']])
    instances_agent_is_in = ds.get_agent_instances(curr_instance['agent_token'])

    trajectory_history = []
    for i in range(inst_idx - stride * (history - 1), inst_idx+1, stride):
        instance = instances_agent_is_in[i]
        pose = np.array(instance['coords'])
        translated_pose = extractor.vis.global_ground_to_local_pixel(current_state, pose)
        trajectory_history.append(np.array([translated_pose[0], translated_pose[1], instance['heading'] - curr_instance['heading']]))
    
    trajectory_future = []
    for i in range(inst_idx + stride, inst_idx + stride * future + 1, stride):
        instance = instances_agent_is_in[i]
        pose = np.array(instance['coords'])
        translated_pose = extractor.vis.global_ground_to_local_pixel(current_state, pose)
        trajectory_future.append(np.array([translated_pose[0], translated_pose[1], instance['heading'] - curr_instance['heading']]))
    
    return image_feature, np.array(trajectory_history), np.array(trajectory_future)

def create_dataset(path, name):
    ds = Dataset()
    ds.load(path + name)

    extractor = TransformerDataProcessor(ds=ds)

    scene = ds.get('scene', ds.list_scenes()[0])
    all_frames = []
    frame_token = scene['first_frame']
    while frame_token:
        all_frames.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']
    
    image_features = []
    trajectory_history = []
    trajectory_future = []

    for frame_idx in tqdm(range(stride*history, len(all_frames) - stride*future, stride)):
        frame_token = all_frames[frame_idx]
        all_instance_tokens, all_instance_indices = extractor.filter_instances(frame_token, stride, history, future)
        num_insts = len(all_instance_tokens)
        
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            inputs = zip(all_instance_tokens, all_instance_indices, [frame_token]*num_insts, [extractor]*num_insts, [ds]*num_insts)
            results = pool.starmap(get_data_for_instance, inputs)
            [image_features.append(feature) for feature, _, _ in results]
            [trajectory_history.append(feature) for _, feature, _ in results]
            [trajectory_future.append(future) for _, _, future in results]
        
    image_features = np.array(image_features)
    trajectory_history = np.array(trajectory_history)
    trajectory_future = np.array(trajectory_future)
    print('img shape', image_features.shape)
    print('history shape', trajectory_history.shape)
    print('future shape', trajectory_future.shape)

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    np.save(DATA_PATH + '/%s_image_feature.npy' % name, image_features)
    np.save(DATA_PATH + '/%s_trajectory_history.npy' % name, trajectory_history)
    np.save(DATA_PATH + '/%s_trajectory_future.npy' % name, trajectory_future)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stride', default=100, help='stride size for saving images. e.g. 10 means save one image per 10 timesteps', type=int)
    parser.add_argument('-p', '--path', default=f"{Path.home()}/MPCLab/dlp-dataset/data/", help='absolute path to JSON files, e.g. ~/dlp-dataset/data/', type=str)
    parser.add_argument('-b', '--before', default=5, help='number of previous observations to store in motion history for input', type=int)
    parser.add_argument('-f', '--future', default=5, help='number of future observations to store as trajectory output', type=int)
    args = parser.parse_args()
    stride = args.stride
    path = args.path
    history = args.before
    future = args.future

    # names = ["DJI_0007", "DJI_0008", "DJI_0009", "DJI_0010", "DJI_0011"]
    names = ["DJI_0012"]
    for name in names:
        create_dataset(path, name)
