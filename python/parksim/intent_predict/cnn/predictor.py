from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.spot_detector.detector import LocalDetector
import torch
from torchvision import transforms
import numpy as np
from parksim.intent_predict.cnn.data_processing.create_dataset import ENTRANCE_TO_PARKING_LOT
from parksim.route_planner.a_star import WaypointsGraph
from parksim.route_planner.a_star import AStarGraph, AStarPlanner
import PIL
import cv2
import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PredictionResponse:
    def __init__(self, all_spot_centers, distribution):
        self.all_spot_centers = all_spot_centers
        self.distribution = distribution
        
    def get_spot_prediction(self):
        index_list = list(range(len(self.distribution)))
        chosen_index = np.random.choice(index_list, p=self.distribution)

        if chosen_index == len(self.all_spot_centers):
            return None
        else: 
            return self.all_spot_centers[chosen_index]
        


class Predictor:
    def __init__(self, resolution=0.1, sensing_limit=20, use_cuda=False):
        # Resolution is distance in meters per pixel
        # sensing_limit: the longest distance to sense along 4 directions (m). The side length of the square = 2*sensing_limit
        self.use_cuda = torch.cuda.is_available()
        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))
        self.resolution = resolution
        self.sensing_limit = sensing_limit

    def load_model(self, waypoints: WaypointsGraph, model_path=ROOT_DIR / 'models/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'):
        self.waypoints = waypoints
        model = SmallRegularizedCNN()
        if self.use_cuda:
            model_state = torch.load(model_path)
        else:
            model_state = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state)
        model.eval()
        if self.use_cuda:
            model = model.cuda()
        self.model = model
            
        
    def predict(self, instance_centric_view, global_position, heading, speed, time_spent_in_lot) -> PredictionResponse:
        # for parking spot in view, predict scores for that spot.
        # normalize scores to get prediction at timestep.
        """
        NOTE:
        See the get_time_spent_in_lot function below for computing the time
        spent in lot
        
        """
        assert self.model != None, "You must call load_model before trying to predict"


        #start = time.time()
        image_features, non_spatial_features, spot_coordinates = self.get_features(instance_centric_view, global_position, heading, speed, time_spent_in_lot)
        #end = time.time()
        #print("Feature Gen Time: ", end - start)
        scores = []
        

        
        
        img_tensors = [transforms.ToTensor()(img_feature) for img_feature in image_features]
        img_tensors = torch.stack(img_tensors, 0)
        non_spatial_tensors = torch.stack([torch.Tensor(non_spatial_feature.astype(np.single)) for non_spatial_feature in non_spatial_features], 0)
        if self.use_cuda:
            img_tensors = img_tensors.cuda()
            non_spatial_tensors = non_spatial_tensors.cuda()
        #start = time.time()
        preds = self.model(img_tensors, non_spatial_tensors)
        #end = time.time()
        #print("Pred Time: ", end - start)
        pred_scores = torch.sigmoid(preds.float())
        #exponentiated_scores = np.exp(scores)
        total_score = torch.sum(pred_scores)
        normalized_scores = pred_scores / total_score
        if self.use_cuda:
            normalized_scores = normalized_scores.cpu()
        normalized_scores = normalized_scores.detach().numpy().reshape(-1,)

        response = PredictionResponse(spot_coordinates, normalized_scores)
        return response
        
        
        index_list = list(range(len(scores)))
        chosen_index = np.random.choice(index_list, p=normalized_scores)

        if chosen_index == len(spot_coordinates):
            response = PredictionResponse(False, None, normalized_scores, spot_coordinates)
        else: 
            response = PredictionResponse(True, spot_coordinates[chosen_index], normalized_scores, spot_coordinates)
        return response
        
    def get_features(self, instance_centric_view, global_position, heading, speed, time_spent_in_lot):
        "Creates (feature list, label) for every spot within sight of the specified agent in the specified frame"
        image_features = []
        non_spatial_features = [] # need some type of way to get the non spatial features
        all_spots = self.get_parking_spots_from_instance(instance_centric_view)
        spot_centers = self.detect_spot_centers(instance_centric_view, global_position, heading, speed)
        ego_speed = speed
        current_global_coords = global_position
        distance_to_entrance = np.linalg.norm(current_global_coords - ENTRANCE_TO_PARKING_LOT)
        
        
        
        for spot_idx, spot in enumerate(all_spots):        
            """Computes features for current spot"""
            astar_dist, astar_dir = self.compute_Astar_dist_dir(global_position, spot_centers[spot_idx], heading)
            
            marked_img = self.label_spot(instance_centric_view, spot)
            img_data = np.array(marked_img)
            image_features.append(img_data)    
            non_spatial_features.append(np.array([[astar_dir, astar_dist, ego_speed, distance_to_entrance, time_spent_in_lot]]))

        unmarked_img = instance_centric_view
        unmarked_img_data = np.array(unmarked_img)
        image_features.append(unmarked_img_data)
        non_spatial_features.append(np.array([[0, 0, ego_speed, distance_to_entrance, time_spent_in_lot]]))
        return image_features, non_spatial_features, spot_centers

    def _get_corners(self, spot):
        return cv2.boxPoints(spot)

    def get_parking_spots_from_instance(self, instance_centric_view):
        img = instance_centric_view
        return self.spot_detector.detect(img)

    def label_spot(self, instance_centric_view, spot):
        instance_view_copy = instance_centric_view.copy()
        corners = self._get_corners(spot)
        img_draw = PIL.ImageDraw.Draw(instance_view_copy)  
        img_draw.polygon(corners, fill ="purple", outline ="purple")
        return instance_view_copy

    def detect_spot_centers(self, instance_centric_view, global_position, heading, speed):
        """
        detect the center of objects visible to the current instance
        """

        img = instance_centric_view
        boxes = self.spot_detector.detect(img) # update later

        box_centers = []

        current_state = np.array([global_position[0], global_position[1], heading, speed])
        for rect in boxes:
            center_pixel = np.array(rect[0])
            center_ground = self.local_pixel_to_global_ground(current_state, center_pixel)
            box_centers.append(center_ground)

        return box_centers

    def local_pixel_to_global_ground(self, current_state, target_coords):
        """
        transform the target coordinate from pixel coordinate in the local inst-centric crop to global ground coordinates
        Note: Accuracy depends on the resolution (self.res)
        current_state: numpy array (x, y, theta, velocity)
        target_coords: numpy array (x, y) in int pixel location
        """
        scaled_local = target_coords * self.resolution
        translation = self.sensing_limit * np.ones(2)

        translated_local = scaled_local - translation

        current_theta = current_state[2]
        R = np.array([[np.cos(current_theta), -np.sin(current_theta)], 
                        [np.sin(current_theta),  np.cos(current_theta)]])

        rotated_local = R @ translated_local

        translated_global = rotated_local + current_state[:2]

        return translated_global

    def compute_Astar_dist_dir(self, vehicle_coords, target_coords: np.ndarray, heading):
        """
        for the A* path to the target spot center, compute the distance and direction

        target_coords: the coordinates of goal. np array (x, y)
        """

        current_vertex_idx = self.waypoints.search(np.array(vehicle_coords))
        spot_vertex_idx = self.waypoints.search(target_coords)

        if spot_vertex_idx == current_vertex_idx:
            astar_dist = 0
            astar_dir = 0
            astar_graph = AStarGraph([])
        else:
            planner = AStarPlanner(
                self.waypoints.vertices[current_vertex_idx], 
                self.waypoints.vertices[spot_vertex_idx])
            astar_graph = planner.solve()

            astar_dist = astar_graph.path_cost()

            path_vector = astar_graph.vertices[1].coords - astar_graph.vertices[0].coords
            heading_vector = np.array([np.cos(heading), np.sin(heading)])

            astar_dir = path_vector @ heading_vector / np.linalg.norm(path_vector) / np.linalg.norm(heading_vector)

        return astar_dist, astar_dir


    def get_time_spent_in_lot(self, ds, agent_token, inst_token):
        SAMPLING_RATE_IN_MINUTES = 0.04 / 60
        instances_agent_is_in = ds.get_agent_instances(agent_token)
        instance_tokens_agent_is_in = [instance['instance_token'] for instance in instances_agent_is_in]
        current_inst_idx = instance_tokens_agent_is_in.index(inst_token)    
        return current_inst_idx * SAMPLING_RATE_IN_MINUTES