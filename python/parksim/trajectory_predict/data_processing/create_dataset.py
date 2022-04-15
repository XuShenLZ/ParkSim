import argparse
import multiprocessing
import numpy as np
import os

from dlp.dataset import Dataset
from math import cos, sin
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

stride = 10
history = 10
future = 10
img_size = 100

_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def get_data_for_instance(inst_token: str, inst_idx: int, frame_token: str, extractor: TransformerDataProcessor, ds: Dataset) -> Tuple[np.array, np.array, np.array]:
    """
    returns image, trajectory_history, and trajectory future for given instance
    """
    img_frame = extractor.vis.plot_frame(frame_token)
    image_feature = extractor.vis.inst_centric(img_frame, inst_token)

    global_intent_pose = extractor.get_intent_pose(inst_token=inst_token, inst_centric_view=image_feature)

    image_feature = extractor.label_target_spot(inst_token, image_feature)

    curr_instance = ds.get('instance', inst_token)
    curr_pose = np.array([curr_instance['coords'][0],
                         curr_instance['coords'][1], curr_instance['heading']])
    rot = np.array([[cos(-curr_pose[2]), -sin(-curr_pose[2])], [sin(-curr_pose[2]), cos(-curr_pose[2])]])
    instances_agent_is_in = ds.get_agent_instances(curr_instance['agent_token'])

    
    local_intent_coords = np.dot(rot, global_intent_pose[:2]-curr_pose[:2])
    local_intent_pose = np.array(
        [local_intent_coords[0], local_intent_coords[1], global_intent_pose[2]-curr_pose[2]])
    local_intent_pose = np.expand_dims(local_intent_pose, axis=0)

    image_history = []
    trajectory_history = []
    for i in range(inst_idx - stride * (history - 1), inst_idx+1, stride):
        instance = instances_agent_is_in[i]
        pos = np.array(instance['coords'])
        translated_pos = np.dot(rot, pos-curr_pose[:2])
        trajectory_history.append(np.array(
            [translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))

        # ======= Uncomment the lines below to generate image history
        img_frame = extractor.vis.plot_frame(instance['frame_token'])
        image_feature = extractor.vis.inst_centric(
            img_frame, instance['instance_token'], curr_pose)
        image_feature = extractor.label_target_spot(
            inst_token, image_feature, curr_pose)
        
        image_history.append(np.asarray(image_feature.resize((img_size, img_size))))
    
    trajectory_future = []
    for i in range(inst_idx + stride, inst_idx + stride * future + 1, stride):
        instance = instances_agent_is_in[i]
        pos = np.array(instance['coords'])
        translated_pos = np.dot(rot, pos-curr_pose[:2])
        trajectory_future.append(np.array([translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))
    
    return np.array(image_history), np.array(trajectory_history), np.array(trajectory_future), local_intent_pose

def create_dataset(path, name, tail_size):
    ds = Dataset()
    ds.load(path + name)

    extractor = TransformerDataProcessor(ds=ds, tail_size=tail_size)

    scene = ds.get('scene', ds.list_scenes()[0])
    all_frames = []
    frame_token = scene['first_frame']
    while frame_token:
        all_frames.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']
    
    image_history = []
    trajectory_history = []
    trajectory_future = []
    intent_pose = []
    cpu_count = os.cpu_count()
    print("CPU COUNT: ", cpu_count)
    for frame_idx in tqdm(range(stride*history, len(all_frames) - stride*future, stride)):
        frame_token = all_frames[frame_idx]
        all_instance_tokens, all_instance_indices = extractor.filter_instances(frame_token, stride, history, future)
        num_insts = len(all_instance_tokens)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            inputs = zip(all_instance_tokens, all_instance_indices, [frame_token]*num_insts, [extractor]*num_insts, [ds]*num_insts)
            results = pool.starmap(get_data_for_instance, inputs)
            [image_history.append(feature) for feature, _, _, _ in results]
            [trajectory_history.append(feature) for _, feature, _, _ in results]
            [trajectory_future.append(feature) for _, _, feature, _ in results]
            [intent_pose.append(feature) for _, _, _, feature in results]
        
    image_history = np.array(image_history)
    trajectory_history = np.array(trajectory_history)
    trajectory_future = np.array(trajectory_future)
    intent_pose = np.array(intent_pose)
    print('img history shape', image_history.shape)
    print('history shape', trajectory_history.shape)
    print('future shape', trajectory_future.shape)
    print('intent pose shape', intent_pose.shape)

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    np.save(os.path.join(DATA_PATH, '%s_image_history.npy' % name), image_history)
    np.save(os.path.join(DATA_PATH,'%s_trajectory_history.npy' % name), trajectory_history)
    np.save(os.path.join(DATA_PATH, '%s_trajectory_future.npy' % name), trajectory_future)
    np.save(os.path.join(DATA_PATH, '%s_intent_pose.npy' % name), intent_pose)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stride', default=10, help='stride size. e.g. 10 means get one data per 10 timesteps', type=int)
    parser.add_argument('-p', '--path', default=f"{Path.home()}/dlp-dataset/data/", help='absolute path to JSON files, e.g. ~/dlp-dataset/data/', type=str)
    parser.add_argument('-b', '--before', default=10, help='number of previous observations to store in motion history for input', type=int)
    parser.add_argument('-f', '--future', default=10, help='number of future observations to store as trajectory output', type=int)
    parser.add_argument('-i', '--img_size', default=100,
                        help='size of the image feature', type=int)
    parser.add_argument('-t', '--tail_size', default=10,
                        help='length of the image tail history', type=int)
    args = parser.parse_args()
    stride = args.stride
    path = args.path
    history = args.before
    future = args.future
    img_size = args.img_size
    tail_size = args.tail_size

    names = ["DJI_" + str(i).zfill(4) for i in range(17, 27)]
    #names = ["DJI_0007", "DJI_0008", "DJI_0009", "DJI_0010", "DJI_0011", "DJI_0012", "DJI_0013", "DJI_0014"]
    #names = ["DJI_0012"]
    for name in names:
        try:
            print(f"Current: {name}")
            create_dataset(path, name, tail_size)
        except Exception as err:
            print(name, "failed")
            print(err)
