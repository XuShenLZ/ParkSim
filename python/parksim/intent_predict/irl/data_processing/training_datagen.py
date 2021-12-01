import argparse
import numpy as np
from numpy.core.fromnumeric import trace
from tqdm import tqdm
import os
import pickle
import traceback

from dlp.dataset import Dataset
from parksim.intent_predict.irl.data_processing.utils import IrlDataProcessor

_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

NUM_FEATURE = 6

def process(stride, path, scene_name):

    # Load drone dataset
    ds = Dataset()
    ds.load(path + scene_name)

    features = []
    labels = []

    processor = IrlDataProcessor(ds)

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

        for inst_token in frame['instances']:
            instance = ds.get('instance', inst_token)
            agent = ds.get('agent', instance['agent_token'])

            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                try:
                    if ds.get_inst_mode(inst_token) != 'incoming':
                        continue

                    spot_centers = processor.detect_center(inst_token, 'spot')
                    if spot_centers == []:
                        continue

                    agent_centers = processor.detect_center(inst_token, 'agent')

                    thres = 3.5

                    ego_speed = instance['speed']

                    # Phi matrix
                    feature = np.zeros((NUM_FEATURE, len(spot_centers)))

                    for center_idx, center_coords in enumerate(spot_centers):

                        local_offset = processor.compute_relative_offset(inst_token, center_coords)

                        astar_dist, astar_dir, astar_graph = processor.compute_Astar_dist_dir(inst_token, center_coords)

                        nearby_agents = 0

                        # speed_list = []
                        for center_agent in agent_centers:
                            dist = astar_graph.dist_to_graph(center_agent)
                            if dist < thres:
                                nearby_agents += 1
                                # agent_speed = processor.get_agent_speed(inst_token, center_agent)
                                # speed_list.append(agent_speed)

                        # max_agent_speed = np.amax(speed_list)
                        # min_agent_speed = np.amin(speed_list)
                        # avg_agent_speed = np.mean(speed_list)

                        feature[0, center_idx] = local_offset[0]
                        feature[1, center_idx] = local_offset[1]
                        feature[2, center_idx] = astar_dist
                        feature[3, center_idx] = astar_dir
                        feature[4, center_idx] = nearby_agents
                        feature[5, center_idx] = ego_speed
                        # feature[6, center_idx] = max_agent_speed
                        # feature[7, center_idx] = min_agent_speed
                        # feature[8, center_idx] = avg_agent_speed

                    label = processor.get_intent_label(inst_token, spot_centers)

                    features.append(feature)
                    labels.append(label)

                except Exception as err:
                    # pass
                    print("==========\nError occured for instance %s" % inst_token)

                    traceback.print_exc()

                
        frame_token = frame['next']

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    with open(DATA_PATH + '/%s_feature.pkl' % scene_name, 'wb') as f:
        pickle.dump(features, f)
    
    with open(DATA_PATH + '/%s_label.pkl' % scene_name, 'wb') as f:
        pickle.dump(labels, f)
    # np.save(DATA_PATH + '/%s_feature.npy' % scene_name, np.asarray(features, dtype=object))
    # np.save(DATA_PATH + '/%s_label.npy' % scene_name, np.asarray(labels, dtype=object))

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
    