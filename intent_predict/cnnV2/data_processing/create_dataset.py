from pathlib import Path
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from dlp.dataset import Dataset
import traceback
from utils import CNNDataProcessor
from tqdm import tqdm
import multiprocessing
from itertools import product

#out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))



_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def get_label_for_index(current_index, target_index):
    return 1 if current_index == target_index else 0

def get_time_spent_in_lot(ds, agent_token, inst_token):
    SAMPLING_RATE_IN_MINUTES = 0.04 / 60
    instances_agent_is_in = ds.get_agent_instances(agent_token)
    instance_tokens_agent_is_in = [instance['instance_token'] for instance in instances_agent_is_in]
    current_inst_idx = instance_tokens_agent_is_in.index(inst_token)    
    return current_inst_idx * SAMPLING_RATE_IN_MINUTES

def get_data_for_instance(inst_token, frame, extractor, ds):
    "Creates (feature list, label) for every spot within sight of the specified agent in the specified frame"
    image_features = []
    non_spatial_features = []
    labels = []
    img_frame = extractor.vis.plot_frame(frame['frame_token'])
    all_spots = extractor.get_parking_spots_from_instance(inst_token, frame)
    spot_centers = extractor.detect_center(inst_token, 'spot')
    selected_spot_index = extractor.get_intent_label(inst_token, spot_centers)
    instance = ds.get('instance', inst_token)
    agent_token = instance['agent_token']
    ego_speed = instance['speed']
    
    ENTRANCE_TO_PARKING_LOT = np.array([20, 80])
    current_global_coords = extractor.get_global_coords(inst_token)
    distance_to_entrance = np.linalg.norm(current_global_coords - ENTRANCE_TO_PARKING_LOT)
    
    
    
    for spot_idx, spot in enumerate(all_spots):
        label = get_label_for_index(spot_idx, selected_spot_index)
        
        """Computes features for current spot"""
        astar_dist, astar_dir, _ = extractor.compute_Astar_dist_dir(inst_token, spot_centers[spot_idx])
        
        marked_img = extractor.label_spot(spot, inst_token, frame)
        img_data = np.array(marked_img)
        image_features.append(img_data)    
        non_spatial_features.append(np.array([[astar_dir, astar_dist, ego_speed, distance_to_entrance, get_time_spent_in_lot(ds, agent_token, inst_token)]]))
        labels.append(label)
    unmarked_img = extractor.vis.inst_centric(img_frame, inst_token)
    no_parking_spot_chosen_index = len(all_spots)
    
    label = get_label_for_index(no_parking_spot_chosen_index, selected_spot_index)
    
    
    
    unmarked_img_data = np.array(unmarked_img)
    image_features.append(unmarked_img_data)
    non_spatial_features.append(np.array([[0, 0, ego_speed, distance_to_entrance, get_time_spent_in_lot(ds, agent_token, inst_token)]]))
    labels.append(label)
    return image_features, non_spatial_features, labels

def filter_instances(ds, instance_tokens):
    filtered_tokens = []
    for inst_token in instance_tokens:
        instance = ds.get('instance', inst_token)
        agent = ds.get('agent', instance['agent_token'])
        if agent['type'] not in {'Pedestrian', 'Undefined'}:
            try:
                if ds.get_inst_mode(inst_token) != 'incoming':
                    continue
                filtered_tokens.append(inst_token)
            except Exception as err:
                print("==========\nError occured for instance %s" % inst_token)
                traceback.print_exc()
    return filtered_tokens


def create_dataset(stride, path, scene_name):
    ds = Dataset()
    ds.load(path + scene_name)
    
    extractor = CNNDataProcessor(ds = ds)
    scene = ds.get('scene', ds.list_scenes()[0])
    all_frames = []
    frame_token = scene['first_frame']
    while frame_token:
        all_frames.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']
    image_features = []
    non_spatial_features = []
    labels = []
    for frame_idx in tqdm(range(0, len(all_frames), stride)):
        frame_token = all_frames[frame_idx]
        frame = ds.get('frame', frame_token)
        all_instance_tokens = filter_instances(ds, frame['instances'])
        
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            inputs = list(product(all_instance_tokens, [frame], [extractor], [ds]))
            results = pool.starmap(get_data_for_instance, inputs)
            [image_features.extend(feature) for feature, _, _ in results]
            [non_spatial_features.extend(feature) for _, feature, _ in results]
            [labels.extend(label) for _, _, label in results]
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        
    image_features = np.array(image_features)
    non_spatial_features = np.array(non_spatial_features)
    labels = np.array(labels)
    np.save(DATA_PATH + '/%s_image_feature.npy' % scene_name, image_features)
    np.save(DATA_PATH + '/%s_non_spatial_feature.npy' % scene_name, non_spatial_features)
    np.save(DATA_PATH + '/%s_label.npy' % scene_name, labels)
        
        
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stride', help='stride size for saving images. e.g. 10 means save one image per 10 timesteps', type=int)
    parser.add_argument('-p', '--path', help='absolute path to JSON files, e.g. ~/dlp-dataset/data/', type=str)
    parser.add_argument('-n', '--name', help='name of the scene, e.g. DJI_0012', type=str)
    args = parser.parse_args()
    stride = args.stride
    path = args.path
    name = args.name
    create_dataset(stride, path, name)