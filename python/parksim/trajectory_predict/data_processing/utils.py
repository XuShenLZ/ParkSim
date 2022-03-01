import numpy as np
from typing import List
import cv2
import PIL
import traceback

from parksim.spot_detector.detector import LocalDetector
from parksim.route_planner.graph import WaypointsGraph
from parksim.route_planner.a_star import AStarPlanner, AStarGraph

from dlp.dataset import Dataset
from dlp.visualizer import SemanticVisualizer

class TransformerDataProcessor(object):
    """
    Extract features for IRL
    """
    def __init__(self, ds: 'Dataset'):
        """
        Instantiate the feature extractor
        """
        self.ds = ds
        self.vis = SemanticVisualizer(ds, steps=10)

        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))
        self.agent_detector = LocalDetector(spot_color_rgb=(255, 255, 0))

        # self.waypoints_graph = WaypointsGraph()
        # self.waypoints_graph.setup_with_vis(self.vis)
    
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
                    if self.ds.get_inst_mode(inst_token) != 'incoming':
                        continue
                    current_inst_idx, max_idx = self.get_instance_index(inst_token, instance['agent_token'])
                    if current_inst_idx < stride * history or max_idx < current_inst_idx + stride * future:
                        continue
                    filtered_tokens.append(inst_token)
                    token_indices.append(current_inst_idx)
                except Exception as err:
                    print("==========\nError occured for instance %s" % inst_token)
                    traceback.print_exc()
        return filtered_tokens, token_indices
