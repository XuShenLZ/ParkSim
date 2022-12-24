from typing import Dict, List
import PIL

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVisualizer

from pathlib import Path

import pickle

import numpy as np

from parksim.vehicle_types import VehicleBody, VehicleConfig
from parksim.route_planner.graph import WaypointsGraph, Vertex
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

import torch

from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor

from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_vision_transformer import TrajectoryPredictorVisionTransformer
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_decoder_intent_cross_attention import TrajectoryPredictorWithDecoderIntentCrossAttention

from dlp.visualizer import SemanticVisualizer
from parksim.intent_predict.cnn.predictor import PredictionResponse, Predictor
import heapq

from parksim.spot_detector.detector import LocalDetector
from parksim.trajectory_predict.intent_transformer.model_utils import generate_square_subsequent_mask

from typing import Tuple
from torch import Tensor
from torchvision import transforms

np.random.seed(39) # ones with interesting cases: 20, 33, 44, 60

# These parameters should all become ROS param for simulator and vehicle
spots_data_path = 'ParkSim/data/spots_data.pickle'
offline_maneuver_path = 'ParkSim/data/parking_maneuvers.pickle'
waypoints_graph_path = 'ParkSim/data/waypoints_graph.pickle'
intent_model_path = 'ParkSim/data/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'
entrance_coords = [14.38, 76.21]

overshoot_ranges = {'pointed_right': [(42, 48), (67, 69), (92, 94), (113, 115), (134, 136), (159, 161), (184, 186), (205, 207), (226, 228), (251, 253), (276, 278), (297, 299), (318, 320), (343, 345)],
                    'pointed_left': [(64, 66), (89, 91), (156, 158), (181, 183), (248, 250), (273, 275), (340, 342)]}

anchor_points = [47, 93, 135, 185, 227, 277, 319, 344] # for now, second spot at the start of a row
anchor_spots = [list(range(21)) + list(range(42, 67)), list(range(21, 42)) + list(range(92, 113)), list(range(67, 92)) + list(range(134, 159)), list(range(113, 134)) + list(range(184, 205)), list(range(159, 184)) + list(range(226, 251)), list(range(205, 226)) + list(range(276, 297)), list(range(251, 276)) + list(range(318, 343)), list(range(297, 318)) + list(range(343, 364))]

north_spot_idx_ranges = [(0, 41), (67, 91), (113, 133), (159, 183), (205, 225), (251, 275), (297, 317)]
spot_y_offset = 5

class RuleBasedSimulator(object):
    def __init__(self, dataset: Dataset, vis: RealtimeVisualizer, anchor_points: List[int]):
        self.dlpvis = DlpVisualizer(dataset)

        self.vis = vis

        self.parking_spaces, self.occupied = self._gen_occupancy()

        self.graph = WaypointsGraph()
        self.graph.setup_with_vis(self.dlpvis)

        self.history = []


        # anchor spots
        self.anchor_points = anchor_points

        # Save
        # with open('waypoints_graph.pickle', 'wb') as f:
        #     data_to_save = {'graph': self.graph, 
        #                     'entrance_coords': entrance_coords}
        #     pickle.dump(data_to_save, f)

        # with open('spots_data.pickle', 'wb') as f:
        #     data_to_save = {'parking_spaces': self.parking_spaces, 
        #                     'overshoot_ranges': overshoot_ranges, 
        #                     'anchor_points': anchor_points,
        #                     'anchor_spots': anchor_spots,
        #                     'north_spot_idx_ranges': north_spot_idx_ranges,
        #                     'spot_y_offset': spot_y_offset}
        #     pickle.dump(data_to_save, f)

        # spawn stuff
        
        spawn_interval_mean = 5 # Mean time for exp distribution
        spawn_interval_min = 2 # Min time for each spawn

        spawn_entering = 1 # number of vehicles to enter
        spawn_exiting = 0 # number of vehicles to exit

        self.spawn_entering_time = sorted(np.random.exponential(spawn_interval_mean, spawn_entering))
        for i in range(spawn_entering):
            self.spawn_entering_time[i] += i * spawn_interval_min

        self.spawn_exiting_time = sorted(np.random.exponential(spawn_interval_mean, spawn_exiting))

        self.num_vehicles = 0
        self.vehicles: List[RuleBasedStanleyVehicle] = []

        self.max_simulation_time = 150

        self.time = 0.0
        self.loops = 0

        # crash detection
        self.did_crash = False
        self.crash_polytopes = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        MODEL_PATH = r'checkpoints/TrajectoryPredictorWithDecoderIntentCrossAttention/lightning_logs/version_1/checkpoints/epoch=52-val_total_loss=0.0458.ckpt'
        self.traj_model = TrajectoryPredictorWithDecoderIntentCrossAttention.load_from_checkpoint(MODEL_PATH)
        self.traj_model.eval().to(self.device)
        self.mode='v1'

        self.intent_extractor = CNNDataProcessor(ds=dataset)
        self.traj_extractor = TransformerDataProcessor(ds=dataset)

        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))

    def _gen_occupancy(self):

        # Spot guide (note: NOT VERTICES) — the i in parking_spaces[i]
        # 0-41 are top row
        # 42-66 are left second row top, 67-91 are left second row bottom
        # 92-112 are right second row top, 113-133 are right second row bottom
        # 134-158 are left third row top, 159-183 are left third row bottom
        # 184-204 are right third row top, 205-225 are right third row bottom
        # 226-250 are left fourth row top, 251-275 are left fourth row bottom
        # 276-296 are right fourth row top, 297-317 are right fourth row bottom
        # 318-342 are left fifth row, 343-363 are right fifth row

        # get parking spaces
        arr = self.dlpvis.parking_spaces.to_numpy()
        # array of tuples of x-y coords of centers of spots
        parking_spaces = np.array([[round((arr[i][2] + arr[i][4]) / 2, 3), round((arr[i][3] + arr[i][9]) / 2, 3)] for i in range(len(arr))])

        scene = self.dlpvis.dataset.get('scene', self.dlpvis.dataset.list_scenes()[0])

        # figure out which parking spaces are occupied
        car_coords = [self.dlpvis.dataset.get('obstacle', o)['coords'] for o in scene['obstacles']]
        # 1D array of booleans — are the centers of any of the cars contained within this spot's boundaries?
        occupied = np.array([any([c[0] > arr[i][2] and c[0] < arr[i][4] and c[1] < arr[i][3] and c[1] > arr[i][9] for c in car_coords]) for i in range(len(arr))])

        return parking_spaces, occupied

    # goes to an anchor point
    # convention: if entering, spot_index is positive, and if exiting, it's negative
    def add_vehicle(self, spot_index: int, vehicle_body: VehicleBody=VehicleBody(), vehicle_config: VehicleConfig=VehicleConfig()):

        # NOTE: These lines are here for now. In the ROS implementation, they will all be in the vehicle node, no the simulator node
        vehicle = RuleBasedStanleyVehicle(vehicle_id=self.num_vehicles, vehicle_body=vehicle_body, vehicle_config=vehicle_config)
        vehicle.load_parking_spaces(spots_data_path=spots_data_path)
        vehicle.load_graph(waypoints_graph_path=waypoints_graph_path)
        vehicle.load_maneuver(offline_maneuver_path=offline_maneuver_path)
        vehicle.set_anchor(going_to_anchor=spot_index>0, spot_index=spot_index, should_overshoot=False)
        vehicle.load_intent_model(model_path=intent_model_path)
        vehicle.start_vehicle()

        self.num_vehicles += 1
        self.vehicles.append(vehicle)
    

    def run(self):
        # while not run out of time and we have not reached the last waypoint yet
        self.circles = None
        self.pred_circles = None
        while self.max_simulation_time >= self.time:


            # clear visualizer
            if self.loops % 10 == 0:
                self.vis.clear_frame()

            
            # spawn vehicles
            if self.spawn_entering_time and self.time > self.spawn_entering_time[0]:
                # self.add_vehicle(np.random.choice(self.anchor_points)) # pick from the anchor points at random
                self.add_vehicle(spot_index=185)
                self.spawn_entering_time.pop(0)
            
            if self.spawn_exiting_time and self.time > self.spawn_exiting_time[0]:
                empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
                chosen_spot = np.random.choice(empty_spots)
                self.add_vehicle(-1 * chosen_spot)
                self.occupied[chosen_spot] = True
                self.spawn_exiting_time.pop(0)

            active_vehicles: Dict[int, RuleBasedStanleyVehicle] = {}
            for vehicle in self.vehicles:
                if not vehicle.is_all_done():
                    active_vehicles[vehicle.vehicle_id] = vehicle

            if not self.spawn_entering_time and not self.spawn_exiting_time and not active_vehicles:
                print("No Active Vehicles")
                break

            # add vehicle states to history
            current_frame_states = []
            for vehicle in self.vehicles:
                current_state_dict = vehicle.get_state_dict()
                current_frame_states.append(current_state_dict)
            self.history.append(current_frame_states)
                
            intent_pred_results = []
            for vehicle_id in active_vehicles:
                vehicle = active_vehicles[vehicle_id]

                vehicle.get_other_info(active_vehicles)
                vehicle.get_central_occupancy(self.occupied)
                vehicle.set_method_to_change_central_occupancy(self.occupied)

                vehicle.solve()
                #result = vehicle.predict_intent(vehicle_id, self.history)
                #intent_pred_results.append(result)

                if self.loops > 100 and self.loops % 10 == 0:

                    intents = vehicle.predict_intent(vehicle_id, self.history)
                    graph = WaypointsGraph()
                    graph.setup_with_vis(self.intent_extractor.vis)
                    best_lanes = self.find_n_best_lanes(
                        [vehicle.state.x.x, vehicle.state.x.y], vehicle.state.e.psi, graph=graph, vis=self.intent_extractor.vis, predictor=vehicle.intent_predictor)

                    distributions, coordinates = self.expand_distribution(intents, best_lanes)

                    top_n = list(zip(distributions, coordinates))
                    top_n.sort(reverse=True)
                    top_n = top_n[:3]

                    output_sequence_length = 9
                    predicted_trajectories = []
                    nth = 0
                    self.circles = []
                    self.pred_circles = []
                    for probability, global_intent_pose in top_n:
                        img, X, intent = self.get_data_for_instance(vehicle, self.traj_extractor, global_intent_pose)

                        with torch.no_grad():
                            if self.mode=='v2':
                                img = img.to(self.device).float()[:, -1]
                            else:
                                img = img.to(self.device).float()
                            X = X.to(self.device).float()
                            intent = intent.to(self.device).float()

                            START_TOKEN = X[:, -1][:, None, :]

                            delta_state = -1 * X[:, -2][:, None, :]
                            y_input = torch.cat([START_TOKEN, delta_state], dim=1).to(self.device)

                            for i in range(output_sequence_length):
                                # Get source mask
                                tgt_mask = generate_square_subsequent_mask(
                                    y_input.size(1)).to(self.device).float()
                                pred = self.traj_model(img, X,
                                                intent, y_input, tgt_mask)
                                next_item = pred[:, -1][:, None, :]
                                # Concatenate previous input with predicted best word
                                y_input = torch.cat((y_input, next_item), dim=1)
                        
                        predicted_trajectories.append(
                            [y_input[:, 1:], intent, probability])
                        
                        if nth == 0:
                            col = (255, 0, 0, 255)
                        elif nth == 1:
                            col = (255, 0, 255, 255)
                        else:
                            col = (9, 121, 105, 255)

                        head = vehicle.state.e.psi
                        inv_rot = np.array([[np.cos(head), -np.sin(head)], [np.sin(head), np.cos(head)]])

                        predicted_intent = intent[0][0].detach().cpu().numpy()
                        local_intent_coords = np.add([vehicle.state.x.x, vehicle.state.x.y], np.dot(inv_rot, predicted_intent))

                        self.circles.append((local_intent_coords, col))

                        preds = []
                        for pt in predicted_trajectories[-1][0][0].detach().cpu().numpy():
                            local_pred_coords = np.add([vehicle.state.x.x, vehicle.state.x.y], np.dot(inv_rot, pt[0:2]))
                            preds.append(local_pred_coords)
                        self.pred_circles.append((preds, col))

                        nth += 1

            # Visualize
            if self.loops % 10 == 0:
                for vehicle in self.vehicles:

                    if vehicle.is_all_done():
                        fill = (0, 0, 0, 255)
                    elif vehicle.is_braking:
                        fill = (255, 0, 0, 255)
                    elif vehicle.parking_flag:
                        fill = (255, 128, 0, 255)
                    else:
                        fill = (0, 255, 0, 255)

                    self.vis.draw_vehicle(state=vehicle.state, fill=fill)
                    if self.circles is not None:
                        for circle, col in self.circles:
                            self.vis.draw_circle(circle, 15, col)
                        for preds, col in self.pred_circles:
                            for pred in preds:
                                self.vis.draw_circle(pred, 3, col)
                    # self.vis.draw_line(points=np.array([vehicle.x_ref, vehicle.y_ref]).T, color=(39,228,245, 193))
                    on_vehicle_text =  str(vehicle.vehicle_id) + ":"
                    on_vehicle_text += "N" if vehicle.priority is None else str(round(vehicle.priority, 3))
                    # self.vis.draw_text([vehicle.state.x.x - 2, vehicle.state.x.y + 2], on_vehicle_text, size=25)
                
            
            self.loops += 1
            self.time += 0.1
            # likelihood_radius = 15
            # for result in intent_pred_results:
            #     distribution = result.distribution
            #     for i in range(len(distribution) - 1):
            #         coords = result.all_spot_centers[i]
            #         prob = format(distribution[i], '.2f')
            #         self.vis.draw_circle(center=coords, radius=likelihood_radius*distribution[i], color=(255,65,255,255))
            #         self.vis.draw_text([coords[0]-2, coords[1]], prob, 15)
    
            self.vis.render()

    def find_n_best_lanes(self, start_coords, global_heading, graph: WaypointsGraph, vis: SemanticVisualizer, predictor: Predictor, n = 3):
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

    def expand_distribution(self, intents: PredictionResponse, lanes: List, n=3):
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

    # TODO: stride was 10 (with 25 FPS, so sampling every 0.4 seconds)
    def get_data_for_instance(self, vehicle: RuleBasedStanleyVehicle, extractor: TransformerDataProcessor, global_intent_pose: np.array, stride: int=4, history: int=10, future: int=10, img_size: int=100) -> Tuple[np.array, np.array, np.array]:
        """
        returns image, trajectory_history, and trajectory future for given instance
        """
        img_transform=transforms.ToTensor()
        image_feature = vehicle.inst_centric_generator.inst_centric(vehicle.vehicle_id, self.history)

        image_feature = self.label_target_spot(vehicle, image_feature)

        curr_pose = np.array([vehicle.state.x.x,
                            vehicle.state.x.y, vehicle.state.e.psi])
        rot = np.array([[np.cos(-curr_pose[2]), -np.sin(-curr_pose[2])], [np.sin(-curr_pose[2]), np.cos(-curr_pose[2])]])
        
        local_intent_coords = np.dot(rot, global_intent_pose[:2]-curr_pose[:2])
        local_intent_pose = np.expand_dims(local_intent_coords, axis=0)

        # determine start index to gather history from
        start_idx = -1
        while start_idx - stride > -stride * history and start_idx - stride >= -len(vehicle.state_hist):
            start_idx -= stride

        image_history = []
        trajectory_history = []
        for i in range(start_idx, 0, stride):
            state = vehicle.state_hist[i]
            pos = np.array([state.x.x, state.x.y])
            translated_pos = np.dot(rot, pos-curr_pose[:2])
            trajectory_history.append(Tensor(
                [translated_pos[0], translated_pos[1], state.e.psi - curr_pose[2]]))

            # ======= Uncomment the lines below to generate image history
            image_feature = vehicle.inst_centric_generator.inst_centric(vehicle.vehicle_id, self.history)
            image_feature = self.label_target_spot(vehicle, image_feature, curr_pose)
            
            image_tensor = img_transform(image_feature.resize((img_size, img_size)))
            image_history.append(image_tensor)
        
        # NOTE: we use [None] to add one dimension to the front
        return torch.stack(image_history)[None], torch.stack(trajectory_history)[None], torch.from_numpy(local_intent_pose)[None]

    def label_target_spot(self, vehicle: RuleBasedStanleyVehicle, inst_centric_view: np.array, center_pose: np.ndarray=None, r=1.25) -> np.array:
        """
        Returns image frame with target spot labeled

        center_pose: If None, the inst_centric_view is assumed to be around the current instance. If a numpy array (x, y, heading) is given, it is the specified center.
        """
        all_spots = self.spot_detector.detect(inst_centric_view)

        # traj = self.vis.dataset.get_future_traj(inst_token)
        if center_pose is None:
            current_state = np.array([vehicle.state.x.x, vehicle.state.x.y, vehicle.state.e.psi])
        else:
            current_state = center_pose

        for spot in all_spots:
            spot_center_pixel = np.array(spot[0])
            spot_center = self.local_pixel_to_global_ground(current_state, spot_center_pixel)
            # TODO: not going to use future trajectory since we don't have access to it
            # dist = np.linalg.norm(traj[:, 0:2] - spot_center, axis=1)
            dist = np.linalg.norm(current_state[0:2] - spot_center)
            # if np.amin(dist) < r:
            if dist < r:
                inst_centric_view_copy = inst_centric_view.copy()
                corners = self._get_corners(spot)
                img_draw = PIL.ImageDraw.Draw(inst_centric_view_copy)  
                img_draw.polygon(corners, fill ="purple", outline ="purple")
                return inst_centric_view_copy
        
        return inst_centric_view

    def local_pixel_to_global_ground(self, current_state: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
        """
        transform the target coordinate from pixel coordinate in the local inst-centric crop to global ground coordinates

        Note: Accuracy depends on the resolution (self.res)

        `current_state`: numpy array (x, y, theta, ...) in global coordinates
        `target_coords`: numpy array (x, y) in int pixel coordinates
        """
        # TODO: taken from SemanticVisualizer in visualizer.py
        sensing_limit = 20
        res = 0.1

        scaled_local = target_coords * res
        translation = sensing_limit * np.ones(2)

        translated_local = scaled_local - translation

        current_theta = current_state[2]
        R = np.array([[np.cos(current_theta), -np.sin(current_theta)], 
                      [np.sin(current_theta),  np.cos(current_theta)]])

        rotated_local = R @ translated_local

        translated_global = rotated_local + current_state[:2]

        return translated_global

def main():
    # Load dataset
    ds = Dataset()

    home_path = Path.home()
    print('Loading dataset...')
    ds.load(str(home_path / 'dlp-dataset/data/DJI_0012'))
    print("Dataset loaded.")

    vis = RealtimeVisualizer(ds, VehicleBody())

    simulator = RuleBasedSimulator(dataset=ds, vis=vis, anchor_points=anchor_points)

    simulator.run()



if __name__ == "__main__":
    main()