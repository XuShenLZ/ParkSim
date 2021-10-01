from pathlib import Path
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from dlp.dataset import Dataset

from utils import CNNDataProcessor
from tqdm import tqdm
import multiprocessing
from itertools import product

#out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))



_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def get_label_for_index(current_index, target_index):
    return 1 if current_index == target_index else 0

def get_data_for_instance(inst_token, frame, extractor):
    features = []
    labels = []
    img_frame = extractor.vis.plot_frame(frame['frame_token'])
    all_spots = extractor.get_parking_spots_from_instance(inst_token, frame)
    spot_centers = extractor.detect_center(inst_token, 'spot')
    selected_spot_index = extractor.get_intent_label(inst_token, spot_centers)
    for spot_idx, spot in enumerate(all_spots):
        label = get_label_for_index(spot_idx, selected_spot_index)
        marked_img = extractor.label_spot(spot, inst_token, frame)
        features.append(np.array(marked_img))
        labels.append(label)
    unmarked_img = extractor.vis.inst_centric(img_frame, inst_token)
    no_parking_spot_chosen_index = len(all_spots)
    
    label = get_label_for_index(no_parking_spot_chosen_index, selected_spot_index)
    
    features.append(np.array(unmarked_img))
    labels.append(label)
    return features, labels

def create_dataset(stride, path, scene_name):
    ds = Dataset()
    ds.load(path + scene_name)
    print(os.cpu_count())
    
    extractor = CNNDataProcessor(ds = ds)
    scene = ds.get('scene', ds.list_scenes()[0])
    all_frames = []
    frame_token = scene['first_frame']
    while frame_token:
        all_frames.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']
    features = []
    labels = []
    for frame_idx in tqdm(range(0, len(all_frames), stride)):
        frame_token = all_frames[frame_idx]
        frame = ds.get('frame', frame_token)
        all_instance_tokens = frame['instances']
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            inputs = list(product(all_instance_tokens, [frame], [extractor]))
            results = pool.starmap(get_data_for_instance, tqdm(inputs, total=len(inputs)))
            [features.extend(feature) for feature, _ in results]
            [labels.extend(label) for _, label in results]
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        
    features = np.array(features)
    labels = np.array(labels)
    np.save(DATA_PATH + '/%s_feature.npy' % scene_name, features)
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