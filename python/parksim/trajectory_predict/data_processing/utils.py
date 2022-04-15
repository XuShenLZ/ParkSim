import numpy as np
from typing import List
import cv2
import PIL
import traceback

from parksim.spot_detector.detector import LocalDetector

from dlp.dataset import Dataset
from dlp.visualizer import SemanticVisualizer

class TransformerDataProcessor(object):
    """
    Extract features for IRL
    """
    def __init__(self, ds: 'Dataset', tail_size=1):
        """
        Instantiate the feature extractor
        """
        self.ds = ds
        self.vis = SemanticVisualizer(ds, steps=tail_size)

        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))
        self.agent_detector = LocalDetector(spot_color_rgb=(255, 255, 0))
    
    def get_instance_index(self, inst_token: str, agent_token: str) -> int:
        """
        get the index of the specified instance relative to the lifetime of its agent
        """
        instances_agent_is_in = self.ds.get_agent_instances(agent_token)
        instance_tokens_agent_is_in = [instance['instance_token'] for instance in instances_agent_is_in]
        return instance_tokens_agent_is_in.index(inst_token), len(instance_tokens_agent_is_in)-1
    
    def filter_instances(self, frame_token: str, stride: int, history: int, future: int) -> List[str]:
        """
        return list of valid (having enough historical data) instances in this frame
        """
        frame = self.ds.get('frame', frame_token)

        filtered_tokens = []
        token_indices = []

        for inst_token in frame['instances']:
            instance = self.ds.get('instance', inst_token)
            agent = self.ds.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                try:
                    if self.ds.get_inst_mode(inst_token) not in ('incoming', 'outgoing'):
                        continue
                    current_inst_idx, max_idx = self.get_instance_index(inst_token, instance['agent_token'])
                    if current_inst_idx < stride * (history-1) or max_idx < current_inst_idx + stride * future:
                        continue
                    filtered_tokens.append(inst_token)
                    token_indices.append(current_inst_idx)
                except Exception as err:
                    print("==========\nError occured for instance %s" % inst_token)
                    traceback.print_exc()
        return filtered_tokens, token_indices

    def _get_corners(self, spot):
        return cv2.boxPoints(spot)
    
    def label_target_spot(self, inst_token: str, inst_centric_view: np.array, center_pose: np.ndarray=None, r=1.25) -> np.array:
        """
        Returns image frame with target spot labeled

        center_pose: If None, the inst_centric_view is assumed to be around the current instance. If a numpy array (x, y, heading) is given, it is the specified center.
        """
        all_spots = self.spot_detector.detect(inst_centric_view)

        traj = self.vis.dataset.get_future_traj(inst_token)
        instance = self.ds.get('instance', inst_token)
        if center_pose is None:
            current_state = np.array([instance['coords'][0], instance['coords'][1], instance['heading']])
        else:
            current_state = center_pose

        for spot in all_spots:
            spot_center_pixel = np.array(spot[0])
            spot_center = self.vis.local_pixel_to_global_ground(current_state, spot_center_pixel)
            dist = np.linalg.norm(traj[:, 0:2] - spot_center, axis=1)
            if np.amin(dist) < r:
                inst_centric_view_copy = inst_centric_view.copy()
                corners = self._get_corners(spot)
                img_draw = PIL.ImageDraw.Draw(inst_centric_view_copy)  
                img_draw.polygon(corners, fill ="purple", outline ="purple")
                return inst_centric_view_copy
        
        return inst_centric_view

    def get_intent_pose(self, inst_token: str, inst_centric_view: np.array, center_pose: np.ndarray = None, r=1.25) -> np.ndarray:
        """
        returns global pose (x, y, heading) coordinates of the intent. 
        
        If the future traj goes inside a spot, use the spot center + vehicle pose as result. If the traj goes outside of the view, use the last visible state as result
        """
        all_spots = self.spot_detector.detect(inst_centric_view)

        traj = self.vis.dataset.get_future_traj(inst_token)
        instance = self.ds.get('instance', inst_token)
        if center_pose is None:
            current_state = np.array(
                [instance['coords'][0], instance['coords'][1], instance['heading']])
        else:
            current_state = center_pose

        for idx in range(len(traj)):
            # If the trajectory goes outside, pick the last step that is still visible
            if not self.vis._is_visible(current_state, traj[idx]):
                return traj[idx-1, 0:3]
            
            # If visible, check whether the vehicle is inside a spot
            for spot in all_spots:
                spot_center_pixel = np.array(spot[0])
                spot_center = self.vis.local_pixel_to_global_ground(
                    current_state, spot_center_pixel)
                dist = np.linalg.norm(traj[idx, 0:2] - spot_center)
                if dist < r:
                    return np.array([spot_center[0], spot_center[1], traj[idx, 2]])

        # Finally, if the program did not return anything, use the last pose along the trajectory
        return traj[-1, 0:3]

