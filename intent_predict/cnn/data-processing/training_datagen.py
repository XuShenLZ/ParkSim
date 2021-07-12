import argparse
import numpy as np
from tqdm import tqdm
import os

from dlp.dataset import Dataset
from utils import PostProcessor

_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def process(stride, path, scene_name):

    # Load drone dataset
    ds = Dataset()
    ds.load(path + scene_name)

    features = []
    labels = []

    processor = PostProcessor(ds)

    scene = ds.get('scene', ds.list_scenes()[0])

    frame_list = []
    frame_token = scene['first_frame']
    while frame_token:
        frame_list.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']
        
    for frame_idx in tqdm(range(0, len(frame_list), stride)):
        frame_token = frame_list[frame_idx]
        
        frame = ds.get('frame', frame_token)

        img_frame = processor.vis.plot_frame(frame_token)

        for inst_token in frame['instances']:
            instance = ds.get('instance', inst_token)
            agent = ds.get('agent', instance['agent_token'])

            if instance['mode']=='moving' and agent['type'] not in {'Pedestrian', 'Undefined'}:
                try:
                    feature, label = processor.gen_feature_label(inst_token, img_frame)
                    features.append(np.asarray(feature))
                    labels.append(label)
                except:
                    pass
                    # print("Error occured for instance %s" % inst_token)
                
        
        frame_token = frame['next']

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

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

    process(stride, path, name)
    