import os
import math
import numpy as np
import argparse

from dlp.dataset import Dataset
import traceback
from parksim.intent_predict.cnn.utils import CNNDataProcessor

#out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))


_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

def get_label_for_index(current_index, target_index):
    return 1 if current_index == target_index else 0


"""
TIME SERIES STUFF HERE
"""
spot_coords_to_id = {}
DISTANCE_THRESHOLD = 0.5

def get_spot_distance(spot_coords, target_coords):
    """
    Returns distance between given spot coordinates and the spot we're checking
    it against
    """
    return math.sqrt((spot_coords[0] - target_coords[0])**2 + (spot_coords[1] - target_coords[1])**2)
def get_spot_id(local_coordinates):
    """
    Returns unique global id of the parking spot specified in local coordinates
    in the instance centric view
    """
    #TODO Does this work across all agents? For example, what if two different
    #spots show up at the same local coordinates for 2 different agents. Won't
    #this fail to see the difference between the spots?
    spot_id = None
    for target_spot_coords in spot_coords_to_id:
        if get_spot_distance(local_coordinates, target_spot_coords) < DISTANCE_THRESHOLD:
            spot_id = spot_coords_to_id[target_spot_coords]
            break
    if not spot_id:
        spot_id = len(spot_coords_to_id)
        spot_coords_to_id[local_coordinates] = spot_id
    return spot_id

def get_parking_spot_ids_in_frame(agent_token, frame, extractor):
    """
    For a given agent token and frame, returns a list of the unique IDs of each
    parking spot in the agent's instance centric view.
    """
    # TODO
    spot_ids_in_frame = []
    all_spots = extractor.get_parking_spots_from_instance(agent_token, frame)
    spot_centers = extractor.detect_center(agent_token, 'spot')
    for spot_center in spot_centers:
        spot_ids_in_frame.append(get_spot_id((spot_center[0], spot_center[1])))
    return spot_ids_in_frame

def get_timestamps_agent_is_in(frames, agent_token):
    """
    For agent token, we want to find the index of the frame in which
    the agent first appears, and the index of the frame in which the
    agent has parked. We then return these two indices.
    """
    
    def frame_contains_agent(frame):
        return agent_token in frame['instances']
    
    first_idx, last_idx = 0, 0
    for i, frame in enumerate(frames):
        if frame_contains_agent(frame):
            first_idx = i
            break
    for i, frame in enumerate(frames[::-1]):
        if frame_contains_agent(frame):
            last_idx = i
            break
    assert first_idx < last_idx, "First index of agent appearance must be less than the last index of appearance."
    return first_idx, last_idx

def get_data_for_agent(agent_token, frames, ds, stride):
    """
    Goes through the agent's path in the parking lot, and construct an array of
    time series data for each spot it encounters along that path.

    Stride represents the step size to take when iterating over frames.

    We will represent the data as a dictionary with keys corresponding to the
    unique IDs of the parking spots the agent sees, and values corresponding to
    another dictionary with keys 'image_feature', 'non_image_feature' and
    'label'. The value for 'image_feature' and 'non_image_feature' is a numpy
    array where axis 0 are time-series data points. In our case, each time
    series data point will have dimensions: (num_time_series_samples,
    feature_dimension).

    Returns a dictionary of dictionaries.
    """
    # TODO
    first_seen_index, parked_index = get_timestamps_agent_is_in(frames, agent_token)
    extractor = CNNDataProcessor(ds = ds)
    agent_data = {}
    for idx in range(first_seen_index, parked_index + 1, stride):
        current_frame = frames[idx]
        parking_spot_ids = get_parking_spot_ids_in_frame(agent_token, current_frame, extractor)
        for parking_spot_id in parking_spot_ids:
            if parking_spot_id not in agent_data:
                agent_data[parking_spot_id] = {'image_feature' : [], 'non_image_feature' : [], 'label' : []}
            image_feature, non_image_feature, label = get_data_for_agent_at_parking_spot_id(agent_token, parking_spot_id, current_frame, ds)
            agent_data[parking_spot_id]['image_feature'].append(image_feature)
            agent_data[parking_spot_id]['non_image_feature'].append(non_image_feature)
            agent_data[parking_spot_id]['label'].append(label)
    return agent_data

def get_data_for_agent_at_parking_spot_id(agent_token, parking_spot_id, frame, ds):
    """
    Returns a tuple (image_feature, non_image_feature, label) where feature represents the feature for
    the agent in the current frame where the parking spot given by
    parking_spot_id is to be colored in. Label is 1 if the agent eventually
    parks at the spot, and 0 otherwise.
    """
    # TODO
    extractor = CNNDataProcessor(ds = ds)
    
    def get_spot():
        all_spots = extractor.get_parking_spots_from_instance(agent_token, frame)
        for spot_idx, spot in enumerate(all_spots):
            local_coords = spot[0]
            if get_spot_id(local_coords) == parking_spot_id:
                return spot_idx, spot
        raise Exception("No spot with this ID was found.")
    
    
    """Image Features"""
    spot_idx, spot = get_spot()
    spot_centers = extractor.detect_center(agent_token, 'spot')
    selected_spot_index = extractor.get_intent_label(agent_token, spot_centers)
    marked_img = extractor.label_spot(spot, agent_token, frame)
    image_feature = np.array(marked_img)
    
    
    """Non Image Features"""
    instance = ds.get('instance', agent_token)
    agent_token = instance['agent_token']
    ego_speed = instance['speed']
    
    ENTRANCE_TO_PARKING_LOT = np.array([20, 80])
    current_global_coords = extractor.get_global_coords(agent_token)
    distance_to_entrance = np.linalg.norm(current_global_coords - ENTRANCE_TO_PARKING_LOT)
    astar_dist, astar_dir, _ = extractor.compute_Astar_dist_dir(agent_token, spot_centers[spot_idx])
    non_image_feature = np.array([[astar_dir, astar_dist, ego_speed, distance_to_entrance, get_time_spent_in_lot(ds, agent_token, agent_token)]])
    
    label = 1 if selected_spot_index == spot_idx else 0
    return image_feature, non_image_feature, label

def pad_feature_data(data):
    """
    List of dictionaries (one for each agent)
        - Keys are the parking spot IDs that the agent saw
        - Values are Dictionaries that correspond to all the time series data
        for that agent





    data: List of dictionaries. Each dictionary corresponds to the data for an
    agent. The keys of this dictionary are the parking spot ids that the agent
    sees over the course of its parking journey. The values are also
    dictionaries of the form: {'image_feature': [ordered list of time series
    image features], 'non_image_feature': [ordered list of time series non image
    features], 'label': [ordered list of time series labels]}

    Will take all of the feature data and find the agent, parking spot
    combination with the maximum time series length.  It will then edit the data
    dictionary in place by zero-padding all other agent, parking spot
    combinations and reassigning the value of data[spot_id]['image_feature'] and
    data[spot_id]['non_image_feature'] in the dictionary, so that later on the
    data can be represented by a matrix/tensor.

    The expected return type is None
    """

    #TODO

    max_time = max(data, key=lambda x: x.size).size
    for time_series in data:
        np.pad(time_series, (0, max_time - time_series.pad), 'constant')
    return np.vstack(max_time)

def combine_agent_data(data):
    """
    data: List of dictionaries. Each dictionary corresponds to the data for an
    agent. The keys of this dictionary are the parking spot ids that the agent
    sees over the course of its parking journey. The values are also
    dictionaries of the form: {'image_feature': [ordered list of time series
    image features], 'non_image_feature': [ordered list of time series non image
    features], 'label': [ordered list of time series labels]}

    Will take all of the data in the dictionary, and concatenate it all into a
    numpy array. Will then return a tuple of (image_features,
    non_image_features, labels) where image_features and non_image_features are
    numpy array of dimensions (sum of number of spots seen by each agent,
    max_time_series_length, feature_shape). Moreover, labels is a numpy array
    with shape (sum of number of spots seen by each agent,
    max_time_series_length).

    Returns (image_features, non_image_features, labels)
    """
    pad_feature_data(data)
    image_features = []
    non_image_features = []
    labels = []
    for spot_id in data:
        data_point = data[spot_id]
        image_features.append(data_point['image_feature'])
        non_image_features.append(data_point['non_image_feature'])
        labels.append(data_point['label'])
    return np.array(image_features), np.array(non_image_features), np.array(labels)



def create_dataset(stride, path, scene_name):
    ds = Dataset()
    ds.load(path + scene_name)



    " Get all agents: "
    all_instance_tokens = get_all_instance_tokens(ds)


    " Get all frames: "

    all_frames = []
    scene = ds.get('scene', ds.list_scenes()[0])
    frame_token = scene['first_frame']
    while frame_token:
        all_frames.append(frame_token)
        frame = ds.get('frame', frame_token)
        frame_token = frame['next']

    " Prepare Data Lists: "
    all_agent_data = []

    " Iterate over agents: "
    for inst_token in all_instance_tokens:
        agent_data = get_data_for_agent(inst_token, all_frames, ds, stride)
        all_agent_data.append(agent_data)

    image_features, non_image_features, labels = combine_agent_data(all_agent_data)

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    np.save(DATA_PATH + '/%s_image_feature.npy' % scene_name, image_features)
    np.save(DATA_PATH + '/%s_non_image_feature.npy' % scene_name, non_image_features)
    np.save(DATA_PATH + '/%s_label.npy' % scene_name, labels)

"""
"""



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

def get_all_instance_tokens(ds):
    scene = ds.get('scene', ds.list_scenes()[0])
    current_frame = ds.get_future_frames(scene['first_frame'],timesteps=1)[0]
    instance_tokens = set()
    while current_frame:
        for inst_token in current_frame['instances']:
            instance_tokens.add(inst_token)
        current_frame = ds.get('frame', current_frame['next'])
    instance_tokens = list(instance_tokens)
    return filter_instances(ds, instance_tokens)

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
