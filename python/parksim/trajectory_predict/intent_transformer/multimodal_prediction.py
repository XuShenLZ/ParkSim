import torch
from torch import Tensor
from torchvision import transforms

from math import cos, sin
import numpy as np
from typing import Tuple

from dlp.dataset import Dataset

from parksim.trajectory_predict.intent_transformer.network import  TrajectoryPredictorWithIntent
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataset

from parksim.intent_predict.cnnV2.network import SmallRegularizedCNN
from parksim.intent_predict.cnnV2.utils import CNNDataset
from parksim.intent_predict.cnnV2.data_processing.utils import CNNDataProcessor
from parksim.intent_predict.cnnV2.predictor import Predictor

def predict_multimodal(ds, traj_model: TrajectoryPredictorWithIntent, intent_model: SmallRegularizedCNN, traj_extractor: TransformerDataProcessor, intent_extractor: CNNDataProcessor, inst_token: str, inst_idx: int, n: int):
    """Given a trajectory prediction model, intent prediction model, and instance,
   predict the top-n most likely trajectories"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    intent_predictor = Predictor(use_cuda=False)
    intent_predictor.waypoints = intent_extractor.waypoints_graph
    intent_predictor.model = intent_model
    instance = ds.get('instance', inst_token)
    img_frame = intent_extractor.vis.plot_frame(instance['frame_token'])
    intent_img = intent_extractor.vis.inst_centric(img_frame, inst_token)
    time_in_lot = intent_predictor.get_time_spent_in_lot(ds, instance['agent_token'], inst_token)
    intents = intent_predictor.predict(intent_img, instance['coords'], instance['heading'], instance['speed'], time_in_lot)

    # TODO: use the waypoints graph to get the non-spot positions and use a heuristic to evaluate the probabilites of those locations
    # for now, just appending 0,0 
    intents.all_spot_centers.append([0,0])

    top_n = list(zip(intents.distribution, intents.all_spot_centers))
    top_n.sort(reverse=True)
    top_n = top_n[:n]

    predicted_trajectories = []
    for probability, global_intent_pose in top_n:
        #TODO: do something with probability?
        img, X, y_label, intent = get_data_for_instance(inst_token, inst_idx, instance['frame_token'], traj_extractor, ds, global_intent_pose)
        with torch.no_grad():
            img = img.to(DEVICE).float()
            X = X.to(DEVICE).float()
            intent = intent.to(DEVICE).float()
            y_label = y_label.to(DEVICE).float()
            y_in = torch.cat((X[-1:], y_label[:-1]))
            y_in = y_in.to(DEVICE).float()
            tgt_mask = traj_model.transformer.generate_square_subsequent_mask(y_in.shape[1]).to(DEVICE).float()
        pred = traj_model(img, X, intent, y_in, tgt_mask)
        predicted_trajectories.append(pred)
    
    return pred

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
    
    #TODO: add an empty dimension as the zero axis to simulate batch size 1
    return torch.stack(image_history), torch.stack(trajectory_history), torch.stack(trajectory_future), torch.from_numpy(local_intent_pose)
