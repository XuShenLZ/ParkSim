from typing import List
import torch
from torch import Tensor
from torchvision import transforms

from math import cos, sin
import numpy as np
from typing import Tuple

from dlp.dataset import Dataset
from dlp.visualizer import SemanticVisualizer

from parksim.trajectory_predict.intent_transformer.models.common_blocks import  BaseTransformerLightningModule
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor
from parksim.trajectory_predict.intent_transformer.model_utils import generate_square_subsequent_mask

from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from parksim.intent_predict.cnn.predictor import PredictionResponse, Predictor

from parksim.route_planner.graph import Vertex, WaypointsGraph
import heapq


def find_n_best_lanes(start_coords, global_heading, graph: WaypointsGraph, vis: SemanticVisualizer, predictor: Predictor, n = 3):
    current_state = np.array(start_coords + [global_heading])
    idx = graph.search(current_state)

    all_lanes = set()
    visited = set()

    fringe: List[Vertex] = [graph.vertices[idx]]
    while len(fringe) > 0:
        v = fringe.pop()
        visited.add(v)

        children, _ = v.get_children()
        if not vis._is_visible(current_state=current_state, target_state=v.coords):
            for child in children:
                if vis._is_visible(current_state=current_state, target_state=child.coords):
                    all_lanes.add(child)
            continue

        for child in children:
            if child not in visited:
                fringe.append(child)

    lanes = []
    for lane in all_lanes:
        astar_dist, astar_dir = predictor.compute_Astar_dist_dir(
            current_state, lane.coords, global_heading)
        heapq.heappush(lanes, (-astar_dir, astar_dist, lane.coords))

    return lanes

def expand_distribution(intents: PredictionResponse, lanes: List, n=3):
    p_minus = intents.distribution[-1]
    n = min(n, len(lanes))

    coordinates = intents.all_spot_centers
    distributions = list(intents.distribution)
    distributions.pop()

    scales = np.linspace(0.9, 0.1, n)
    scales /= sum(scales)
    for i in range(n):
        _, _, coords = heapq.heappop(lanes)
        coordinates.append(coords)
        distributions.append(p_minus * scales[i])

    return distributions, coordinates
            


def predict_multimodal(ds, traj_model: BaseTransformerLightningModule, intent_model: SmallRegularizedCNN, traj_extractor: TransformerDataProcessor, intent_extractor: CNNDataProcessor, inst_token: str, inst_idx: int, n: int, mode='v2'):
    """Given a trajectory prediction model, intent prediction model, and instance,
   predict the top-n most likely trajectories"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        intent_predictor = Predictor(use_cuda=True)
    else:
        intent_predictor = Predictor(use_cuda=False)
    intent_predictor.waypoints = intent_extractor.waypoints_graph
    intent_predictor.model = intent_model
    instance = ds.get('instance', inst_token)
    img_frame = intent_extractor.vis.plot_frame(instance['frame_token'])
    intent_img = intent_extractor.vis.inst_centric(img_frame, inst_token)
    time_in_lot = intent_predictor.get_time_spent_in_lot(ds, instance['agent_token'], inst_token)
    intents = intent_predictor.predict(intent_img, instance['coords'], instance['heading'], instance['speed'], time_in_lot)

    # TODO: use the waypoints graph to get the non-spot positions and use a heuristic to evaluate the probabilites of those locations
    graph = WaypointsGraph()
    graph.setup_with_vis(intent_extractor.vis)
    best_lanes = find_n_best_lanes(
        instance['coords'], instance['heading'], graph=graph, vis=intent_extractor.vis, predictor=intent_predictor)
    # for now, just appending 0,0 

    distributions, coordinates = expand_distribution(intents, best_lanes)

    top_n = list(zip(distributions, coordinates))
    top_n.sort(reverse=True)
    top_n = top_n[:n]

    output_sequence_length = 9
    predicted_trajectories = []
    for probability, global_intent_pose in top_n:
        #TODO: do something with probability?
        img, X, y_label, intent = get_data_for_instance(inst_token, inst_idx, instance['frame_token'], traj_extractor, ds, global_intent_pose)
        with torch.no_grad():
            if mode=='v2':
                img = img.to(DEVICE).float()[:, -1]
            else:
                img = img.to(DEVICE).float()
            X = X.to(DEVICE).float()
            intent = intent.to(DEVICE).float()
            y_label = y_label.to(DEVICE).float()

            # X is trajectory history, relative to current state
            # START_TOKEN is most recent item in history (e.g. usually [0, 0, 0])
            # delta_state is change in state from second-most recent to most recent
            START_TOKEN = X[:, -1][:, None, :]

            delta_state = -1 * X[:, -2][:, None, :]
            # initially, y_input is if you applied the same most recent change in state again
            y_input = torch.cat([START_TOKEN, delta_state], dim=1).to(DEVICE)
            # y_input = START_TOKEN

            # predict next best output_sequence_length steps in MPC-style fashion
            for i in range(output_sequence_length):
                # Get source mask
                tgt_mask = generate_square_subsequent_mask(
                    y_input.size(1)).to(DEVICE).float()
                pred = traj_model(img, X,
                                  intent, y_input, tgt_mask)
                # next_item is predicted next best action
                next_item = pred[:, -1][:, None, :]
                # Concatenate previous input with predicted best word
                y_input = torch.cat((y_input, next_item), dim=1)
                # y_input[:, i+1, :] = pred[:, i, :]

            # now y_input has a bunch of predicted next states
                
            # return y_input[:, 1:]

            # img = img.to(DEVICE).float()
            # X = X.to(DEVICE).float()
            # intent = intent.to(DEVICE).float()
            # y_label = y_label.to(DEVICE).float()
            # # y_in = torch.cat((X[-1:], y_label[:-1]))
            # y_in = y_label.to(DEVICE).float()
            # tgt_mask = traj_model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(DEVICE).float()
        # pred = traj_model(img, X, intent, y_in, tgt_mask)
        # y_input comprehension is so we don't include current state in prediction
        predicted_trajectories.append(
            [y_label, y_input[:, 1:], intent, probability])
        # predicted_trajectories.append(
        #         [y_label, y_input, intent, probability])
        # predicted_trajectories.append(
        #     [y_label, pred, intent, probability])
    
    return predicted_trajectories, intent_img

def get_data_for_instance(inst_token: str, inst_idx: int, frame_token: str, extractor: TransformerDataProcessor, ds: Dataset, global_intent_pose: np.array, stride: int=10, history: int=10, future: int=10, img_size: int=100) -> Tuple[np.array, np.array, np.array]:
    """
    returns image, trajectory_history, and trajectory future for given instance
    """
    img_transform=transforms.ToTensor()
    img_frame = extractor.vis.plot_frame(frame_token)
    image_feature = extractor.vis.inst_centric(img_frame, inst_token)

    image_feature = extractor.label_target_spot(inst_token, image_feature)

    curr_instance = ds.get('instance', inst_token)
    curr_pose = np.array([curr_instance['coords'][0],
                         curr_instance['coords'][1], curr_instance['heading']])
    rot = np.array([[cos(-curr_pose[2]), -sin(-curr_pose[2])], [sin(-curr_pose[2]), cos(-curr_pose[2])]])
    instances_agent_is_in = ds.get_agent_instances(curr_instance['agent_token'])
    inst_idx = None
    for i, inst in enumerate(instances_agent_is_in):
        if inst['instance_token'] == inst_token:
            inst_idx = i
            break
    
    local_intent_coords = np.dot(rot, global_intent_pose[:2]-curr_pose[:2])
    local_intent_pose = np.expand_dims(local_intent_coords, axis=0)

    image_history = []
    trajectory_history = []
    for i in range(inst_idx - stride * (history - 1), inst_idx+1, stride):
        instance = instances_agent_is_in[i]
        pos = np.array(instance['coords'])
        translated_pos = np.dot(rot, pos-curr_pose[:2])
        trajectory_history.append(Tensor(
            [translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))

        # ======= Uncomment the lines below to generate image history
        img_frame = extractor.vis.plot_frame(instance['frame_token'])
        image_feature = extractor.vis.inst_centric(
            img_frame, instance['instance_token'])
        image_feature = extractor.label_target_spot(
            inst_token, image_feature, curr_pose)
        
        image_tensor = img_transform(image_feature.resize((img_size, img_size)))
        image_history.append(image_tensor)
    
    trajectory_future = []
    for i in range(inst_idx + stride, inst_idx + stride * future + 1, stride):
        instance = instances_agent_is_in[i]
        pos = np.array(instance['coords'])
        translated_pos = np.dot(rot, pos-curr_pose[:2])
        trajectory_future.append(Tensor([translated_pos[0], translated_pos[1], instance['heading'] - curr_pose[2]]))
    
    # NOTE: we use [None] to add one dimension to the front
    return torch.stack(image_history)[None], torch.stack(trajectory_history)[None], torch.stack(trajectory_future)[None], torch.from_numpy(local_intent_pose)[None]
