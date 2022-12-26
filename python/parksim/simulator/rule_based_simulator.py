from typing import Dict, List
import PIL

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVisualizer

from pathlib import Path

import pickle
import os
import glob
import csv
import dataclasses

import numpy as np
import torch
from scipy.io import savemat
from parksim.pytypes import VehicleState

from parksim.vehicle_types import VehicleBody, VehicleConfig, VehicleTask
from parksim.route_planner.graph import WaypointsGraph, Vertex
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

from parksim.controller.stanley_controller import StanleyController, normalize_angle
from parksim.controller_types import StanleyParams
from parksim.spot_nn.spot_nn import SpotNet
from parksim.spot_nn.feature_generator import SpotFeatureGenerator

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

from parksim.utils.spline import calc_spline_course
import pyomo.environ as pyo

# These parameters should all become ROS param for simulator and vehicle
spots_data_path = '/ParkSim/data/spots_data.pickle'
offline_maneuver_path = '/ParkSim/data/parking_maneuvers.pickle'
waypoints_graph_path = '/ParkSim/data/waypoints_graph.pickle'
intent_model_path = '/ParkSim/data/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'

class RuleBasedSimulator(object):
    def __init__(self, dataset: Dataset, vis: RealtimeVisualizer, params):

        self.params = params

        self.timer_period = 0.1

        self.blocked_spots = [42, 43, 44, 45, 46, 47, 48, 49, 64, 65, 66, 67, 68, 69, 92, 93, 94, 110, 111, 112, 113, 114, 115, 134, 135, 136, 156, 157, 158, 159, 160, 161, 184, 185, 186, 202, 203, 204, 205, 206, 207, 226, 227, 228, 248, 249, 250, 251, 252, 253, 276, 277, 278, 294, 295, 256, 297, 298, 299, 318, 319, 320, 340, 341, 342, 343, 344, 345] # Spots to be blocked in advance: 3 left and 3 right spaces of each row, except right spaces of right row, since doesn't unpark into an aisle
        self.entrance_coords = [14.38, 76.21]

        self.spawn_entering = params.spawn_entering
        self.spawn_exiting = params.spawn_exiting
        self.y_bound_to_resume_spawning = 70
        self.spawn_interval_mean = params.spawn_interval_mean # (s)

        self.spots_data_path = '/ParkSim/data/spots_data.pickle'
        self.agents_data_path = params.agents_data_path

        self.use_existing_agents = params.use_existing_agents
        self.use_existing_obstacles = params.use_existing_obstacles 

        self.write_log = False
        self.log_path = '/ParkSim/vehicle_log'

        self.dlpvis = DlpVisualizer(dataset)
        self.vis = vis
        self.should_visualize = params.should_visualize

        self.graph = WaypointsGraph()
        self.graph.setup_with_vis(self.dlpvis)

        # Save data to offline files
        # with open(str(Path.home()) + '/parksim/data/waypoints_graph.pickle', 'wb') as f:
        #     data_to_save = {'graph': self.graph, 
        #                     'entrance_coords': self.entrance_coords}
        #     pickle.dump(data_to_save, f)

        # Clean up the log folder if needed
        if self.write_log:
            log_dir_path = str(Path.home()) + self.log_path

            if not os.path.exists(log_dir_path):
                os.mkdir(log_dir_path)
            log_files = glob.glob(log_dir_path+'/*.log')
            for f in log_files:
                os.remove(f)

        # Parking Spaces
        self.parking_spaces, self.occupied = self._gen_occupancy()
        for idx in self.blocked_spots:
            self.occupied[idx] = True

        # Agents
        self._gen_agents()

        # Spawning
        self.spawn_entering_time = list(np.random.exponential(self.spawn_interval_mean, self.spawn_entering))
        self.spawn_entering_time_cumsum = list(np.cumsum(np.array(self.spawn_entering_time)))
        self.spawn_exiting_time = list(np.random.exponential(self.spawn_interval_mean, self.spawn_exiting))

        self.last_enter_id = None
        self.last_enter_state = VehicleState()
        self.last_entering_vehicle_left_entrance = False
        self.keep_spawn_entering = True

        self.start_time = 0

        self.last_enter_time = self.start_time
        self.last_exit_time = self.start_time

        self.vehicles = []
        self.num_vehicles = 0
        self.vehicle_non_idle_times = {}

        self.sim_is_running = True

        self.num_vehicles = 0
        self.vehicles: List[RuleBasedStanleyVehicle] = []

        self.max_simulation_time = 2000

        self.time = 0.0
        self.loops = 0   

        self.vehicle_features = {}
        self.vehicle_ids_entered = []

        if params.use_nn or params.train_nn:
            self.feature_generator = params.feature_generator

        self.queue_length = 0

        # uncomment this to regenerate data used to generate features (1/3)
        # self.discretization = {}
        # self.traj_len = {}
        # self.last_pt = {}

        # ev charging
        self.ev_simulation = params.ev_simulation
        self.charging_spots = [39, 40, 41] # TODO: put this in a yaml

        # prediction controller

        self.history = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        MODEL_PATH = r'checkpoints/TrajectoryPredictorWithDecoderIntentCrossAttention/lightning_logs/version_1/checkpoints/epoch=52-val_total_loss=0.0458.ckpt'
        self.traj_model = TrajectoryPredictorWithDecoderIntentCrossAttention.load_from_checkpoint(MODEL_PATH)
        self.traj_model.eval().to(self.device)
        self.mode='v1'

        self.intent_extractor = CNNDataProcessor(ds=dataset)
        self.traj_extractor = TransformerDataProcessor(ds=dataset)

        self.spot_detector = LocalDetector(spot_color_rgb=(0, 255, 0))

        self.intent_circles = []
        self.traj_pred_circles = []

        self.model = None
        self.solver = pyo.SolverFactory('ipopt')

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
        car_coords = [self.dlpvis.dataset.get('obstacle', o)['coords'] for o in scene['obstacles']] if self.use_existing_obstacles else []
        # 1D array of booleans — are the centers of any of the cars contained within this spot's boundaries?
        occupied = np.array([any([c[0] > arr[i][2] and c[0] < arr[i][4] and c[1] < arr[i][3] and c[1] > arr[i][9] for c in car_coords]) for i in range(len(arr))])

        return parking_spaces, occupied

    def _gen_agents(self):
        home_path = str(Path.home())
        with open(home_path + self.agents_data_path, 'rb') as f:
            self.agents_dict = pickle.load(f)

            # determine when vehicles will unpark, so that we don't assign cars to park in those spots before other vehicles appear to unpark
            # self.unparking times[spot_index] = time a vehicle will unpark from there
            self.total_vehicle_count = len(self.agents_dict)
            self.unparking_times = {}
            max_init_time = -1
            for agent in self.agents_dict:
                if self.agents_dict[agent]["task_profile"][0]["name"] == "UNPARK":
                    self.unparking_times[self.agents_dict[agent]["task_profile"][0]["target_spot_index"]] = self.agents_dict[agent]["init_time"]
                max_init_time = max(max_init_time, self.agents_dict[agent]["init_time"])
            if self.use_existing_agents:
                self.spawn_interval_mean = max_init_time / len(self.agents_dict) * 2 # times 2 since entering and exiting both come in with that mean

    # goes to an anchor point
    # convention: if entering, spot_index is positive, and if exiting, it's negative
    def add_vehicle(self, spot_index: int=None, vehicle_body: VehicleBody=VehicleBody(), vehicle_config: VehicleConfig=VehicleConfig(), vehicle_id: int=None, for_nn: bool=False):
        if not for_nn:
            # Start vehicle indexing from 1
            self.num_vehicles += 1
            if vehicle_id is None:
                vehicle_id = self.num_vehicles

        if self.use_existing_agents:
            agent_dict = self.agents_dict[vehicle_id]

            vehicle_body.w = agent_dict["width"]
            vehicle_body.l = agent_dict["length"]

        controller_params = StanleyParams(dt=self.timer_period)
        controller = StanleyController(control_params=controller_params, vehicle_body=vehicle_body, vehicle_config=vehicle_config)
        motion_predictor = StanleyController(control_params=controller_params, vehicle_body=vehicle_body, vehicle_config=vehicle_config)

        vehicle = RuleBasedStanleyVehicle(
            vehicle_id=vehicle_id, 
            vehicle_body=dataclasses.replace(vehicle_body), 
            vehicle_config=dataclasses.replace(vehicle_config), 
            controller=controller,
            motion_predictor=motion_predictor,
            electric_vehicle=self.ev_simulation
            )

        vehicle.load_parking_spaces(spots_data_path=spots_data_path)
        vehicle.load_graph(waypoints_graph_path=waypoints_graph_path)
        vehicle.load_maneuver(offline_maneuver_path=offline_maneuver_path)
        vehicle.load_intent_model(model_path=intent_model_path)

        task_profile = []

        if not self.use_existing_agents:
            if spot_index >= 0:
                cruise_task = VehicleTask(
                    name="CRUISE", v_cruise=5, target_spot_index=spot_index)
                park_task = VehicleTask(name="PARK", target_spot_index=spot_index)
                task_profile = [cruise_task, park_task]

                state = VehicleState()
                state.x.x = self.entrance_coords[0] - vehicle_config.offset
                state.x.y = self.entrance_coords[1]
                state.e.psi = - np.pi/2

                vehicle.set_vehicle_state(state=state)
                
                self.vehicle_ids_entered.append(vehicle_id)
            else:
                unpark_task = VehicleTask(name="UNPARK")
                cruise_task = VehicleTask(
                    name="CRUISE", v_cruise=5, target_coords=np.array(self.entrance_coords))
                task_profile = [unpark_task, cruise_task]

                vehicle.set_vehicle_state(spot_index=abs(spot_index))
        else:
            raw_tp = agent_dict["task_profile"]
            for task in raw_tp:
                if task["name"] == "IDLE":
                    if "end_time" in task:
                        task_profile.append(VehicleTask(name="IDLE", end_time=task["end_time"]))
                    else:
                        task_profile.append(VehicleTask(name="IDLE", duration=task["duration"]))
                elif task["name"] == "PARK":
                    task_profile.append(VehicleTask(name="PARK", target_spot_index=task["target_spot_index"]))
                    self.occupied[task["target_spot_index"]] = True
                elif task["name"] == "UNPARK":
                    task_profile.append(VehicleTask(name="UNPARK", target_spot_index=task["target_spot_index"]))
                elif task["name"] == "CRUISE":
                    if "target_coords" in task:
                        task_profile.append(VehicleTask(name="CRUISE", v_cruise=task["v_cruise"], target_coords=task["target_coords"]))
                    else:
                        task_profile.append(VehicleTask(name="CRUISE", v_cruise=task["v_cruise"], target_spot_index=task["target_spot_index"]))

            # determine which vehicles entered from the entrance for stats purposes later
            for task in raw_tp:
                if task["name"] == "PARK" and "init_coords" in agent_dict and agent_dict["init_coords"][1] > self.y_bound_to_resume_spawning:
                    vehicle.spot_index = task["target_spot_index"]
                    self.vehicle_ids_entered.append(vehicle_id)
                elif task["name"] == "UNPARK":
                    break

            if "init_spot" in agent_dict:
                init_spot = agent_dict["init_spot"]
                init_heading = agent_dict["init_heading"]
                vehicle.set_vehicle_state(spot_index=init_spot, heading=init_heading)
            else:
                agent_state = VehicleState()
                # vehicles tend to enter from the same place they exit, which can cause overlaps/jams
                if agent_dict["init_coords"][1] > self.y_bound_to_resume_spawning:
                    agent_state.x.x = agent_dict["init_coords"][0] - 3
                else:
                    agent_state.x.x = agent_dict["init_coords"][0]
                agent_state.x.y = agent_dict["init_coords"][1]
                agent_state.e.psi = agent_dict["init_heading"]
                agent_state.v.v = agent_dict["init_v"]
                vehicle.set_vehicle_state(state=agent_state)

        vehicle.set_task_profile(task_profile=task_profile)
        vehicle.execute_next_task()

        if not for_nn:
            self.vehicle_non_idle_times[vehicle_id] = 0
            if (spot_index and spot_index >= 0) or (self.use_existing_agents and self.vehicle_entering(vehicle_id)):
                self.last_enter_state = vehicle.state
                self.last_enter_id = vehicle_id
                self.last_entering_vehicle_left_entrance = False

            self.vehicles.append(vehicle)
        else:
            # uncomment this to regenerate data used to generate features (2/3)
            # squares_traveled = set()
            # for i in range(len(vehicle.controller.x_ref)):
            #     x = vehicle.controller.x_ref[i]
            #     y = vehicle.controller.y_ref[i]
            #     sq = self.feature_generator.coord_to_heatmap_square(vehicle.controller.x_ref[i], vehicle.controller.y_ref[i])
            #     squares_traveled.add(sq)
            #     if x > 7.71 and x <= 28.53 and y > 61.4 and y <= 68.51:
            #         squares_traveled.add(sq - 1)
            #     elif x > 76.54 and x <= 83.82 and y > 61.4 and y <= 68.51:
            #         squares_traveled.add(sq + 1)
            #     elif x > 7.71 and x <= 76.54 and y > 6.48 and y <= 68.51:
            #         squares_traveled.add(sq - 1)
            #         squares_traveled.add(sq + 1)
            #     elif x > 83.82 and x <= 138.42 and y > 6.48 and y <= 68.51:
            #         squares_traveled.add(sq - 1)
            #         squares_traveled.add(sq + 1)
            # self.discretization[spot_index] = squares_traveled
            # self.traj_len[spot_index] = vehicle.controller.get_ref_length()
            # self.last_pt[spot_index] = (vehicle.controller.x_ref[-1], vehicle.controller.y_ref[-1])

            return vehicle

    def try_spawn_entering(self):
        current_time = self.time

        active_vehicles = []
        for vehicle in self.vehicles:
            if not vehicle.is_all_done():
                active_vehicles.append(vehicle)

        # if self.spawn_entering_time and current_time - self.last_enter_time > self.spawn_entering_time[0]:
        if self.spawn_entering_time and current_time > self.spawn_entering_time_cumsum[0]:
            empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
            chosen_spot = self.params.choose_spot(self, empty_spots, active_vehicles)
            self.add_vehicle(chosen_spot)
            self.occupied[chosen_spot] = True
            self.spawn_entering_time.pop(0)
            self.spawn_entering_time_cumsum.pop(0)

            self.last_enter_time = current_time
            self.last_enter_id = self.num_vehicles
            self.keep_spawn_entering = False

    def try_spawn_exiting(self):
        current_time = self.time

        if self.spawn_exiting_time and current_time - self.last_exit_time > self.spawn_exiting_time[0]:
            empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
            chosen_spot = np.random.choice(empty_spots)
            self.add_vehicle(-1 * chosen_spot)
            self.occupied[chosen_spot] = True
            self.spawn_exiting_time.pop(0)

            self.last_exit_time = current_time

    def try_spawn_existing(self):
        current_time = self.time - self.start_time

        # determine vehicles to add — one entering + any others
        vehicles_to_add = []
        earliest_entering_vehicle = None
        for agent in self.agents_dict:
            if self.agents_dict[agent]["init_time"] < current_time: # time check
                if not self.vehicle_entering(agent): # if not entering, always spawn
                    vehicles_to_add.append(agent)
                elif self.last_entering_vehicle_left_entrance:
                    if earliest_entering_vehicle is None: # first entering vehicle we've checked
                        earliest_entering_vehicle = (agent, self.agents_dict[agent]["init_time"])
                    elif earliest_entering_vehicle[1] > self.agents_dict[agent]["init_time"]: # current earliest entering vehicle entered later than this vehicle
                        earliest_entering_vehicle = (agent, self.agents_dict[agent]["init_time"])
        if earliest_entering_vehicle is not None:
            vehicles_to_add.append(earliest_entering_vehicle[0])

        for agent in vehicles_to_add:
            # DJI 25
            # if agent == 181:
            #     continue
            # change task profile to park in nn spot if the vehicle is entering from the entrance
            if not self.params.use_existing_entrances and self.vehicle_entering(agent):
                last_unpark = -1 # index of the most recent unparking
                already_parked = None # None if the vehicle has not parked yet, else it's the spot index we've assigned
                new_tp = self.agents_dict[agent]["task_profile"] # task profile we will change
                
                # look for a parking task so we can change it
                for i, task in enumerate(new_tp):
                    # if see an unpark
                    if task["name"] == "UNPARK":
                        # if haven't already determined new parking spot, set this as the most recent unpark
                        if already_parked is None:
                            last_unpark = i
                        else: # if already determined new parking spot, this is an unpark after a park, so set the spot we are unparking from to our new spot
                            new_tp[i]["target_spot_index"] = already_parked
                            last_unpark = i
                    # if see a park we need to change
                    elif (not self.is_electric_vehicle(agent) and task["name"] == "PARK" and i > 0) \
                        or (self.is_electric_vehicle(agent) and task["name"] == "PARK" and i > 0 and "target_spot_index" not in task):

                        # collect arguments to choose spot
                        active_vehicles = []
                        for vehicle in self.vehicles:
                            if not vehicle.is_all_done():
                                active_vehicles.append(vehicle)
                        empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i] and (i not in self.unparking_times or self.unparking_times[i] < self.time)]

                        # choose new spot
                        new_spot_index = self.params.choose_spot(self, empty_spots, active_vehicles)

                        # create tasks and insert into task profile
                        cruise_speed = max([t["v_cruise"] for t in new_tp[last_unpark + 1:i] if t["name"] == "CRUISE"])
                        cruise_task = {"name": "CRUISE", "v_cruise": cruise_speed, "target_spot_index": new_spot_index}
                        park_task = {"name": "PARK", "target_spot_index": new_spot_index}
                        new_tp = new_tp[:last_unpark + 1] + new_tp[i + 1:]
                        new_tp.insert(last_unpark + 1, cruise_task)
                        new_tp.insert(last_unpark + 2, park_task)

                        # state management
                        already_parked = new_spot_index
                        self.occupied[new_spot_index] = True
                    # if parking in an ev charging spot, remember last charging spot
                    elif self.is_electric_vehicle(agent) and task["name"] == "PARK" and i > 0 and "target_spot_index" in task:
                        already_parked = task["target_spot_index"]
                self.agents_dict[agent]["task_profile"] = new_tp

            self.add_vehicle(vehicle_id=agent)

        for added in vehicles_to_add:
            del self.agents_dict[added]

    def vehicle_entering(self, vehicle_id):
        return "init_coords" in self.agents_dict[vehicle_id] and self.agents_dict[vehicle_id]["init_coords"][1] > self.y_bound_to_resume_spawning

    def is_electric_vehicle(self, vehicle_id):
        return "ev_charging" in self.agents_dict[vehicle_id] and self.agents_dict[vehicle_id]["ev_charging"]

    def run(self):
        # uncomment this to regenerate data used to generate features (3/3)
        # for ve in range(364):
        #     self.add_vehicle(ve, for_nn=True)
        # print(self.discretization)
        # dat = {}
        # dat["trajectory_squares"] = self.discretization
        # dat["trajectory_length"] = self.traj_len
        # dat["last_waypoint"] = self.last_pt
        # with open(str(Path.home()) + "/parksim/python/parksim/spot_nn/create_features_data.pickle", 'wb') as file:
        #     pickle.dump(dat, file)
        #     print("done")
        #     file.close()

        if self.write_log:
            # write logs
            log_dir_path = str(Path.home()) + self.log_path
            if not os.path.exists(log_dir_path):
                os.mkdir(log_dir_path)

        # TODO: params
        # self.set_up_model()

        # determine when to end sim for use_existing_agents
        if self.use_existing_agents:
            last_existing_init_time = max([self.agents_dict[agent]["init_time"] for agent in self.agents_dict])

        # while not run out of time and we have not reached the last waypoint yet
        while self.max_simulation_time >= self.time:

            if self.should_visualize:
                if not self.vis.is_running():
                    self.vis.render()
                    continue

                # clear visualizer
                self.vis.clear_frame()

            if self.sim_is_running:
                if not self.use_existing_agents:
                    self.queue_length = max(sum([t < self.time for t in self.spawn_entering_time_cumsum]) - 1, 0) # since the spawn entering time is not a multiple of 0.1, this includes the spawning vehicle
                    if self.keep_spawn_entering:
                        self.try_spawn_entering()
                    self.try_spawn_exiting()
                else:
                    self.queue_length = max(sum([self.agents_dict[v]["init_time"] < self.time and self.vehicle_entering(v) for v in self.agents_dict]) - 1, 0)
                    self.try_spawn_existing()

            active_vehicles: Dict[int, RuleBasedStanleyVehicle] = {}
            for vehicle in self.vehicles:
                if not vehicle.is_all_done():
                    active_vehicles[vehicle.vehicle_id] = vehicle

            if not self.use_existing_agents and not self.spawn_entering_time and not self.spawn_exiting_time and not active_vehicles:
                # print("No Active Vehicles")
                break
            elif self.use_existing_agents and self.time > last_existing_init_time and not active_vehicles:
                break

            # If vehicle left entrance area, start spawning another one
            if self.last_enter_state.x.y < self.y_bound_to_resume_spawning or self.last_enter_id and self.last_enter_id not in active_vehicles:
                self.keep_spawn_entering = True
                if not self.last_entering_vehicle_left_entrance:
                    self.last_entering_vehicle_left_entrance = True

            # add vehicle states to history
            current_frame_states = {}
            for vehicle in self.vehicles:
                current_state_dict = vehicle.get_state_dict()
                current_frame_states[vehicle.vehicle_id] = current_state_dict
            self.history.append(current_frame_states)
                
            # intent_pred_results = []

            # obtain states for all vehicles first, then solve for all vehicles (mimics ROS)
            for vehicle_id in active_vehicles:
                vehicle = active_vehicles[vehicle_id]

                vehicle.get_other_info(active_vehicles)
                vehicle.get_central_occupancy(self.occupied)
                vehicle.set_method_to_change_central_occupancy(self.occupied)

            for vehicle_id in active_vehicles:
                vehicle = active_vehicles[vehicle_id]
                
                if self.write_log:
                    with open(log_dir_path + "/vehicle_%d.log" % vehicle.vehicle_id, 'a') as f:
                        f.writelines(str(self.vehicle_non_idle_times[vehicle.vehicle_id]))
                        vehicle.logger.clear()

                    # write velocity data
                    velocities = []
                    st = vehicle.state_hist[0].t
                    for s in vehicle.state_hist:
                        velocities.append([s.t - st, s.v.v])
                    savemat(str(Path.home()) + "/ParkSim/vehicle_log/DJI_0022/simulated_vehicle_" + str(vehicle.vehicle_id) + ".mat", {"velocity": velocities})

                if vehicle.current_task != "IDLE":
                    self.vehicle_non_idle_times[vehicle_id] += self.timer_period

                if (self.params.use_nn or self.params.train_nn) and vehicle_id in self.vehicle_ids_entered and vehicle_id not in self.vehicle_features:
                    self.vehicle_features[vehicle_id] = self.feature_generator.generate_features(vehicle.spot_index, [active_vehicles[id] for id in active_vehicles], self.spawn_interval_mean, self.queue_length)

                if self.sim_is_running:
                    vehicle.solve(time=self.time)
                elif self.write_log and len(vehicle.logger) > 0:
                    # write logs
                    log_dir_path = str(Path.home()) + self.log_path
                    if not os.path.exists(log_dir_path):
                        os.mkdir(log_dir_path)
                    
                    with open(log_dir_path + "/vehicle_%d.log" % vehicle.vehicle_id, 'a') as f:
                        f.writelines('\n'.join(vehicle.logger))
                        vehicle.logger.clear()

                # ========== For real-time prediction only
                # result = vehicle.predict_intent()
                # intent_pred_results.append(result)
                # ===========

                if self.loops > 4:

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
                    self.intent_circles = []
                    self.traj_pred_circles = []
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

                        predicted_intent = intent[0][0]
                        local_intent_coords = np.add([vehicle.state.x.x, vehicle.state.x.y], np.dot(inv_rot, predicted_intent))

                        self.intent_circles.append((local_intent_coords, col))

                        preds = []
                        for pt in predicted_trajectories[0][0]:
                            local_pred_coords = np.add([vehicle.state.x.x, vehicle.state.x.y], np.dot(inv_rot, pt[0:2]))
                            preds.append(local_pred_coords)
                        self.traj_pred_circles.append((preds, col))

                        preds = np.array(preds)
                        # spline
                        cxs, cys, cyaws, _, _ = calc_spline_course(preds[:, 0], preds[:, 1], ds=self.timer_period)
                        xref = np.array(list(zip(cxs[:10], cys[:10], np.zeros(10), np.zeros(10))))

                        # TODO: use params
                        """
                        _, feas, x2, u2, _ = self.solve_model(N=xref.shape[0], x0=np.array([vehicle.state.x.x, vehicle.state.x.y, vehicle.state.v.v, vehicle.state.e.psi]), xref=xref)
                        """

                        # TODO: problem is that NN predicts too slow, so mpc wants to curve trajectory all the time (even when driving straight) so that we slow down
                        # solve optimal control problem
                        _, feas, xOpt, uOpt, _ = self.solve_cftoc(P=np.diag([1, 1, 0, 0]), Q=np.diag([1, 1, 0, 0]), R=np.zeros((2, 2)), N=xref.shape[0], x0=np.array([vehicle.state.x.x, vehicle.state.x.y, vehicle.state.v.v, vehicle.state.e.psi]), uL=np.array([vehicle.vehicle_config.a_min, vehicle.vehicle_config.delta_min]), uU=np.array([vehicle.vehicle_config.a_max, vehicle.vehicle_config.delta_max]), xref=xref, vehicle=vehicle)

                        # get control (is control 0 problematic because the first xref is usually just continuing at the same speed in the same direction? not sure)
                        control = uOpt[:, 0]

                        # display predictions from optimal control problem
                        mpc_preds = xOpt[[0, 1]].T
                        # self.traj_pred_circles.append((mpc_preds, col))

                        # display "steering wheel"
                        control_mag = control[0] * 3
                        control_dir = vehicle.state.e.psi + control[1]
                        control_dot = [vehicle.state.x.x + (control_mag * np.cos(control_dir)), vehicle.state.x.y + (control_mag * np.sin(control_dir))]
                        self.intent_circles.append((control_dot, col))

                        nth += 1
        
            self.loops += 1
            self.time += self.timer_period

            if self.should_visualize:

                # label charging spots
                if self.ev_simulation:
                    for spot in self.charging_spots:
                        self.vis.draw_text([self.parking_spaces[spot][0] - 1, self.parking_spaces[spot][1] + 2], 'C', size=40)

                # Visualize
                for vehicle in self.vehicles:

                    if not vehicle.ev:
                        if vehicle.is_all_done():
                            fill = (0, 0, 0, 255)
                        elif vehicle.is_braking:
                            fill = (255, 0, 0, 255)
                        elif vehicle.current_task in ["PARK", "UNPARK"]:
                            fill = (255, 128, 0, 255)
                        else:
                            fill = (0, 255, 0, 255)
                    else:
                        if vehicle.is_all_done():
                            fill = (0, 0, 0, 255)
                        elif vehicle.ev_charging_state == 0:
                            fill = (255, 0, 0, 255)
                        elif vehicle.ev_charging_state == 1:
                            fill = (255, 128, 0, 255)
                        else:
                            fill = (0, 255, 0, 255)

                    self.vis.draw_vehicle(state=vehicle.state, fill=fill)
                    
                    if self.intent_circles is not None:
                        for circle, col in self.intent_circles:
                            self.vis.draw_circle(circle, 15, col)
                        for preds, col in self.traj_pred_circles:
                            for pred in preds:
                                self.vis.draw_circle(pred, 3, col)

                    on_vehicle_text =  str(vehicle.vehicle_id)
                    self.vis.draw_text([vehicle.state.x.x - 2, vehicle.state.x.y + 2], on_vehicle_text, size=25)
                    
                # ========== For real-time prediction only
                # likelihood_radius = 15
                # for result in intent_pred_results:
                #     distribution = result.distribution
                #     for i in range(len(distribution) - 1):
                #         coords = result.all_spot_centers[i]
                #         prob = format(distribution[i], '.2f')
                #         self.vis.draw_circle(center=coords, radius=likelihood_radius*distribution[i], color=(255,65,255,255))
                #         self.vis.draw_text([coords[0]-2, coords[1]], prob, 15)
                # ===========
        
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

    # stride was 10 on dlp example (with 25 FPS, so sampling every 0.4 seconds)
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

            image_feature = vehicle.inst_centric_generator.inst_centric(vehicle.vehicle_id, self.history)
            image_feature = self.label_target_spot(vehicle, image_feature, curr_pose)
            
            image_tensor = img_transform(image_feature.resize((img_size, img_size)))
            image_history.append(image_tensor)
        
        return torch.stack(image_history)[None], torch.stack(trajectory_history)[None], torch.from_numpy(local_intent_pose)[None]

    def label_target_spot(self, vehicle: RuleBasedStanleyVehicle, inst_centric_view: np.array, center_pose: np.ndarray=None, r=1.25) -> np.array:
        """
        Returns image frame with target spot labeled
        center_pose: If None, the inst_centric_view is assumed to be around the current instance. If a numpy array (x, y, heading) is given, it is the specified center.
        """
        all_spots = self.spot_detector.detect(inst_centric_view)

        if center_pose is None:
            current_state = np.array([vehicle.state.x.x, vehicle.state.x.y, vehicle.state.e.psi])
        else:
            current_state = center_pose

        for spot in all_spots:
            spot_center_pixel = np.array(spot[0])
            spot_center = self.local_pixel_to_global_ground(current_state, spot_center_pixel)
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

    """

    def set_up_model(self, vehicle_body: VehicleBody=VehicleBody(), vehicle_config: VehicleConfig=VehicleConfig()):
        self.model = pyo.ConcreteModel()
        self.model.N = pyo.Param(mutable=True, initialize=10)
        self.model.nx = 4 # x, y, v, psi
        self.model.nu = 2 # acceleration, delta (steering)
        
        # length of finite optimization problem:
        self.model.tIDX = pyo.Set( initialize= range(self.model.N.value+1), ordered=True )  
        self.model.xIDX = pyo.Set( initialize= range(self.model.nx), ordered=True )
        self.model.uIDX = pyo.Set( initialize= range(self.model.nu), ordered=True )
        
        # these are 2d arrays:
        self.model.P = np.diag([1, 1, 0, 0])
        self.model.Q = np.diag([1, 1, 0, 0])
        self.model.R = np.zeros((self.model.nu, self.model.nu))
        self.model.x0 = pyo.Param(mutable=True, initialize=np.array([0, 0, 0, 0]))
        self.model.xref = pyo.Param(mutable=True, initialize=np.zeros((4, 11)))
        
        # Create state and input variables trajectory:
        self.model.x = pyo.Var(self.model.xIDX, self.model.tIDX)
        self.model.u = pyo.Var(self.model.uIDX, self.model.tIDX)

        self.model.uL = np.array([vehicle_config.a_min, vehicle_config.delta_min])
        self.model.uU = np.array([vehicle_config.a_max, vehicle_config.delta_max])
    
        #Objective:
        def objective_rule(model):
            costX = 0.0
            costU = 0.0
            costTerminal = 0.0
            for t in model.tIDX:
                for i in model.xIDX:
                    for j in model.xIDX:
                        if t < model.N.value:
                            costX += (model.x[i, t] - model.xref.value[i, t]) * model.Q[i, j] * (model.x[j, t] - model.xref.value[j, t]) 
            for t in model.tIDX:
                for i in model.uIDX:
                    for j in model.uIDX:
                        if t < model.N.value:
                            costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
            for i in model.xIDX:
                for j in model.xIDX:               
                    costTerminal += (model.x[i, model.N.value] - model.xref.value[i, model.N.value]) * model.P[i, j] * (model.x[j, model.N.value] - model.xref.value[j, model.N.value])
            return costX + costU + costTerminal
        
        self.model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
        
        # Constraints:
        self.model.init_const = pyo.Constraint(self.model.xIDX, rule=lambda model, i: self.model.x[i, 0] == self.model.x0.value[i])
        
        self.model.bike_const_x = pyo.Constraint(self.model.tIDX, rule=lambda model, t: self.model.x[0, t+1] == self.model.x[0, t] + self.timer_period * (self.model.x[2, t] * pyo.cos(self.model.x[3, t])) if t < self.model.N.value else pyo.Constraint.Skip)
        self.model.bike_const_y = pyo.Constraint(self.model.tIDX, rule=lambda model, t: self.model.x[1, t+1] == self.model.x[1, t] + self.timer_period * (self.model.x[2, t] * pyo.sin(self.model.x[3, t])) if t < self.model.N.value else pyo.Constraint.Skip)
        self.model.bike_const_v = pyo.Constraint(self.model.tIDX, rule=lambda model, t: self.model.x[2, t+1] == self.model.x[2, t] + self.timer_period * (self.model.u[0, t]) if t < self.model.N.value else pyo.Constraint.Skip)
        self.model.bike_const_psi = pyo.Constraint(self.model.tIDX, rule=lambda model, t: self.model.x[3, t+1] == self.model.x[3, t] + self.timer_period * (self.model.x[2, t] * pyo.tan(self.model.u[1, t]) / vehicle_body.wb) if t < self.model.N.value else pyo.Constraint.Skip)

        self.model.input_const_l = pyo.Constraint(self.model.uIDX, self.model.tIDX, rule=lambda model, i, t: self.model.u[i, t] <= self.model.uU[i] if t < self.model.N.value else pyo.Constraint.Skip)
        self.model.input_const_u = pyo.Constraint(self.model.uIDX, self.model.tIDX, rule=lambda model, i, t: self.model.u[i, t] >= self.model.uL[i] if t < self.model.N.value else pyo.Constraint.Skip)

    def solve_model(self, N, x0, xref):
        self.model.N = N
        self.model.x0 = x0
        self.model.xref = np.vstack((x0, xref)).T

        results = self.solver.solve(self.model)
        
        if str(results.solver.termination_condition) == "optimal":
            feas = True
        else:
            feas = False
                
        xOpt = np.asarray([[self.model.x[i,t]() for i in self.model.xIDX] for t in self.model.tIDX]).T
        uOpt = np.asarray([self.model.u[:,t]() for t in self.model.tIDX]).T
        
        JOpt = self.model.cost()
        
        return [self.model, feas, xOpt, uOpt, JOpt]

    """

    def solve_cftoc(self, P, Q, R, N, x0, uL, uU, xref, vehicle: RuleBasedStanleyVehicle):
        model = pyo.ConcreteModel()
        model.N = N
        model.nx = 4 # x, y, v, psi
        model.nu = 2 # acceleration, delta (steering)
        
        # length of finite optimization problem:
        model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )  
        model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
        model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
        
        # these are 2d arrays:
        model.P = P
        model.Q = Q
        model.R = R
        model.xref = np.vstack((x0, xref)).T
        
        # Create state and input variables trajectory:
        model.x = pyo.Var(model.xIDX, model.tIDX)
        model.u = pyo.Var(model.uIDX, model.tIDX)
    
        #Objective:
        def objective_rule(model):
            costX = 0.0
            costU = 0.0
            costTerminal = 0.0
            for t in model.tIDX:
                for i in model.xIDX:
                    for j in model.xIDX:
                        if t < model.N:
                            costX += (model.x[i, t] - model.xref[i, t]) * model.Q[i, j] * (model.x[j, t] - model.xref[j, t]) 
            for t in model.tIDX:
                for i in model.uIDX:
                    for j in model.uIDX:
                        if t < model.N:
                            costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
            for i in model.xIDX:
                for j in model.xIDX:               
                    costTerminal += (model.x[i, model.N] - model.xref[i, model.N]) * model.P[i, j] * (model.x[j, model.N] - model.xref[j, model.N])
            return costX + costU + costTerminal
        
        model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
        
        # Constraints:
        model.init_const = pyo.Constraint(model.xIDX, rule=lambda model, i: model.x[i, 0] == x0[i])
        
        model.bike_const_x = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[0, t+1] == model.x[0, t] + self.timer_period * (model.x[2, t] * pyo.cos(model.x[3, t])) if t < N else pyo.Constraint.Skip)
        model.bike_const_y = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[1, t+1] == model.x[1, t] + self.timer_period * (model.x[2, t] * pyo.sin(model.x[3, t])) if t < N else pyo.Constraint.Skip)
        model.bike_const_v = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[2, t+1] == model.x[2, t] + self.timer_period * (model.u[0, t]) if t < N else pyo.Constraint.Skip)
        model.bike_const_psi = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[3, t+1] == model.x[3, t] + self.timer_period * (model.x[2, t] * pyo.tan(model.u[1, t]) / vehicle.vehicle_body.wb) if t < N else pyo.Constraint.Skip)

        model.input_const_l = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] <= uU[i] if t < N else pyo.Constraint.Skip)
        model.input_const_u = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] >= uL[i] if t < N else pyo.Constraint.Skip)

        solver = pyo.SolverFactory('ipopt')
        results = solver.solve(model)
        
        if str(results.solver.termination_condition) == "optimal":
            feas = True
        else:
            feas = False
                
        xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
        uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T
        
        JOpt = model.cost()
        
        return [model, feas, xOpt, uOpt, JOpt]

"""
Change these parameters to run tests using the neural network
"""
class RuleBasedSimulatorParams():
    def __init__(self):
        self.seed = 0

        self.num_simulations = 1 # number of simulations run (e.g. times started from scratch)

        self.ev_simulation = False # electric vehicle (Soomin's data) sim?

        self.use_existing_agents = False # replay video data
        self.agents_data_path = '/ParkSim/data/agents_data_0012.pickle'

        # should we replace where the agents park?
        self.use_existing_entrances = True # have vehicles park in spots that they parked in real life

        # don't use existing agents
        self.spawn_entering_fn = lambda: 1 
        self.spawn_exiting_fn = lambda: 0
        self.spawn_interval_mean_fn = lambda: 0 # (s)

        self.use_existing_obstacles = True # able to park in "occupied" spots from dataset? False if yes, True if no

        self.load_existing_net = False # generate a new net form scratch (and overwrite model.pickle) or use the one stored at self.spot_model_path
        self.use_nn = False # pick spots using NN or not (irrelevant if self.use_existing_entrances is True)
        self.train_nn = False # train NN or not
        self.should_visualize = self.num_simulations == 1 # display simulator or not

        if self.use_nn or self.train_nn:

            # before changing model, don't forget to set: spot selection, loss function
            self.spot_model_path = '/Parksim/python/parksim/spot_nn/selfish_model.pickle'
            self.losses_csv_path = '/parksim/python/parksim/spot_nn/losses.csv' # where losses are stored

            # load net
            if self.load_existing_net:
                self.net = torch.load(str(Path.home()) + self.spot_model_path)
            else:
                self.net = SpotNet()

            self.feature_generator = SpotFeatureGenerator()

    # run simulations, including training the net (if necessary) and saving/printing any results
    def run_simulations(self, ds, vis):

        if self.use_nn or self.train_nn:

            if os.path.isfile(str(Path.home()) + self.spot_model_path) and not self.load_existing_net:
                print("error: would be overwriting exisitng net. please rename existing net or set self.load_existing_net to True.")
                quit()

        losses = []
        average_times = []

        for i in range(self.num_simulations):
            self.spawn_entering = self.spawn_entering_fn()
            self.spawn_exiting = self.spawn_exiting_fn()
            self.spawn_interval_mean = self.spawn_interval_mean_fn()
            simulator = RuleBasedSimulator(dataset=ds, vis=vis, params=self)
            if not self.use_existing_agents:
                print('Experiment ' + str(i) + ': ' + str(self.spawn_entering) + ' entering, ' + str(self.spawn_exiting) + ' exiting, ' + str(self.spawn_interval_mean) + ' spawn interval mean')
            simulator.run()
            if self.use_existing_agents:
                print('Experiment ' + str(i) + ': ' + str(len(simulator.vehicle_ids_entered)) + ' entering, ' + str(simulator.total_vehicle_count) + ' total, ' + str(simulator.spawn_interval_mean) + ' spawn interval mean')

            if self.use_nn or self.train_nn:

                total_loss = self.update_net(simulator)
                losses.append(total_loss / len(simulator.vehicle_ids_entered) if total_loss is not None else 0)
                average_times.append(sum([simulator.vehicle_non_idle_times[i] for i in simulator.vehicle_ids_entered]) / len(simulator.vehicle_ids_entered))
                print('Results: ' + (str(losses[-1]) if total_loss is not None else 'N/A') + ' average loss, ' + str(average_times[-1]) + ' average entering time')

        if self.use_nn or self.train_nn:
        
            with open(str(Path.home()) + self.losses_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Simulation", "Loss", "Average Entering Time"])
                for i in range(len(losses)):
                    writer.writerow([i, losses[i], average_times[i]])

    # spot selection algorithm
    def choose_spot(self, simulator: RuleBasedSimulator, empty_spots: List[int], active_vehicles: List[RuleBasedStanleyVehicle]):
        if self.use_nn:
            # return min([spot for spot in empty_spots], key=lambda spot: self.net(self.feature_generator.generate_features(spot, active_vehicles, simulator.spawn_interval_mean, simulator.queue_length)))
            nth_smallest = 0
            sorted_spots = sorted(empty_spots, key=lambda spot: self.net(self.feature_generator.generate_features(spot, active_vehicles, simulator.spawn_interval_mean, simulator.queue_length)))
            return sorted_spots[nth_smallest]
            # if self.num_vehicles % 2 == 0:
            #     chosen_spot = min([spot for spot in empty_spots], key=lambda spot: self.spot_net(SpotFeatureGenerator.generate_features(self.add_vehicle(spot_index=spot, for_nn=True), active_vehicles, self.spawn_interval_mean, simulator.queue_length)))
            # else:
            #     chosen_spot = min([spot for spot in empty_spots], key=lambda spot: self.vanilla_net(SpotFeatureGenerator.generate_features(self.add_vehicle(spot_index=spot, for_nn=True), active_vehicles, self.spawn_interval_mean, simulator.queue_length)))
        else:
            return 190
            """
            r = 0
            if r < 0.4:
                return np.random.choice(empty_spots)
            elif r < 0.8:
                return np.random.choice([i for i in empty_spots if (i >= 46 and i <= 66) or (i >= 70 and i <= 91) or (i >= 137 and i <= 158) or (i >= 162 and i <= 183)])
            else:
                return min([spot for spot in empty_spots], key=lambda spot: np.linalg.norm([simulator.entrance_coords[0] - simulator.parking_spaces[spot][0], simulator.entrance_coords[1] - simulator.parking_spaces[spot][1]]))
            """

    # target function for neural net
    def target(self, simulator: RuleBasedSimulator, vehicle_id: int, vehicles_included: List[int]):
        net_discount = 0.9
        # return torch.FloatTensor([sum([(net_discount ** i) * simulator.vehicle_non_idle_times[vehicles_included[i]] for i in range(len(vehicles_included))])])
        return torch.FloatTensor([simulator.vehicle_non_idle_times[vehicle_id]])

    # update network and return loss
    def update_net(self, simulator: RuleBasedSimulator):
        num_vehicles_included = 5
        if self.train_nn:
            total_loss = 0
            # for v in simulator.vehicle_ids_entered: # IDs
            #     loss = self.net.update(simulator.vehicle_features[v], self.target(simulator, v))
            for i, v in enumerate(simulator.vehicle_ids_entered[:-num_vehicles_included]):
                loss = self.net.update(simulator.vehicle_features[v], self.target(simulator, v, simulator.vehicle_ids_entered[i:i+num_vehicles_included]))
                total_loss += loss.detach().numpy()
            self.save_net()
            return total_loss
        return None

    # save NN parameters to disk
    def save_net(self):
        torch.save(self.net, str(Path.home()) + self.spot_model_path)

def main():
    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    print('Loading dataset...')
    ds.load(home_path + '/dlp-dataset/data/DJI_0012')
    print("Dataset loaded.")

    vis = RealtimeVisualizer(ds, VehicleBody())

    params = RuleBasedSimulatorParams()

    if params.seed is not None:
        np.random.seed(params.seed)
        
    params.run_simulations(ds, vis)
    

if __name__ == "__main__":
    main()