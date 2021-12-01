import numpy as np
from typing import List
import pickle

from parksim.spot_detector.detector import LocalDetector
from parksim.route_planner.graph import WaypointsGraph
from parksim.route_planner.a_star import AStarPlanner, AStarGraph

from dlp.dataset import Dataset
from dlp.visualizer import SemanticVisualizer

np.random.seed(1)

class IrlDataProcessor(object):
    """
    Extract features for IRL
    """
    def __init__(self, ds: 'Dataset'):
        """
        Instantiate the feature extractor
        """
        self.ds = ds
        self.vis = SemanticVisualizer(ds, steps=0)

        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))
        self.agent_detector = LocalDetector(spot_color_rgb=(255, 255, 0))

        self.waypoints_graph = WaypointsGraph()
        self.waypoints_graph.setup_with_vis(self.vis)

    def detect_center(self, inst_token, obj_type):
        """
        detect the center of objects visible to the current instance
        """
        assert obj_type in ['spot', 'agent']

        instance = self.ds.get('instance', inst_token)
        img_frame = self.vis.plot_frame(instance['frame_token'])
        img = self.vis.inst_centric(img_frame, inst_token)

        if obj_type == 'spot':
            boxes = self.spot_detector.detect(img)
        elif obj_type == 'agent':
            boxes = self.agent_detector.detect(img)

        box_centers = []

        current_state = np.array([instance['coords'][0], instance['coords'][1], instance['heading'], instance['speed']])
        for rect in boxes:
            center_pixel = np.array(rect[0])
            center_ground = self.vis.local_pixel_to_global_ground(current_state, center_pixel)
            box_centers.append(center_ground)

        return box_centers

    def compute_relative_offset(self, inst_token, target_coords: np.ndarray):
        """
        compute the offset along (x, y) direction in vehicle coordinate system
        """
        instance = self.ds.get('instance', inst_token)
        global_offset = target_coords - np.array(instance['coords'])
        
        R = np.array([[np.cos(-instance['heading']), -np.sin(-instance['heading'])], 
                      [np.sin(-instance['heading']),  np.cos(-instance['heading'])]])

        return R @ global_offset

    def compute_Astar_dist_dir(self, inst_token, target_coords: np.ndarray):
        """
        for the A* path to the target spot center, compute the distance and direction

        target_coords: the coordinates of goal. np array (x, y)
        """
        instance = self.ds.get('instance', inst_token)

        current_vertex_idx = self.waypoints_graph.search(np.array(instance['coords']))
        spot_vertex_idx = self.waypoints_graph.search(target_coords)

        if spot_vertex_idx == current_vertex_idx:
            astar_dist = 0
            astar_dir = 0
            astar_graph = AStarGraph([])
        else:
            planner = AStarPlanner(
                self.waypoints_graph.vertices[current_vertex_idx], 
                self.waypoints_graph.vertices[spot_vertex_idx])
            astar_graph = planner.solve()

            astar_dist = astar_graph.path_cost()

            path_vector = astar_graph.vertices[1].coords - astar_graph.vertices[0].coords
            heading_vector = np.array([np.cos(instance['heading']), np.sin(instance['heading'])])

            astar_dir = path_vector @ heading_vector / np.linalg.norm(path_vector) / np.linalg.norm(heading_vector)

        return astar_dist, astar_dir, astar_graph

    def get_agent_speed(self, inst_token, agent_coords: np.ndarray):
        """
        search the agent with the provided center coords and return its speed
        """
        instance = self.ds.get('instance', inst_token)
        frame = self.ds.get('frame', instance['frame_token'])

        closet_inst_dist = np.inf
        closet_inst_token = None
        for _inst_token in frame['instances']:
            _instance = self.ds.get('instance', _inst_token)

            # Only check vehicle agents
            _agent = self.ds.get('agent', _instance['agent_token'])
            if _agent['type'] in {'Pedestrian', 'Undefined'}:
                continue

            dist = np.linalg.norm(np.array(_instance['coords']) - agent_coords)
            if dist < closet_inst_dist:
                closet_inst_token = _inst_token
                closet_inst_dist = dist

        return self.ds.get('instance', closet_inst_token)['speed']

    def get_intent_label(self, inst_token, spot_centers: List[np.ndarray], r=1.25):
        """
        compute the intent label as the prob distribution over visible spots

        r: The radius to determine whether the car is inside a spot or not
        """
        traj = self.vis.dataset.get_future_traj(inst_token)
        
        # The default label is a uniform distribution
        label = np.ones(len(spot_centers)) / len(spot_centers)

        # Check whether the trajectory will enter one of the detected spots
        for idx, center_coords in enumerate(spot_centers):
            dist = np.linalg.norm(traj[:, 0:2] - center_coords, axis=1)

            if np.amin(dist) < r:
                label = np.zeros(len(spot_centers))
                label[idx] = 1.

        return label

class IrlDataLoader(object):
    """
    Dataset Loader for Irl Data
    """
    def __init__(self, file_path, normalize=True, shuffle=True):
        """
        Load the generated IRL dataset
        """
        with open('%s_feature.pkl' % file_path, 'rb') as f:
            self.features_raw = pickle.load(f)

        with open('%s_label.pkl' % file_path, 'rb') as f:
            self.labels = pickle.load(f)

        if normalize:
            concat_feature = np.concatenate(self.features_raw, axis=1)
            norm_factor = np.linalg.norm(concat_feature, axis=1)

            self.features = [x / norm_factor[:, None] for x in self.features_raw]
        else:
            self.features = self.features_raw

        if shuffle:
            idx = np.random.permutation(len(self.features))
            self.features = [self.features[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label