from typing import Dict, List

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
from parksim.route_planner.graph import WaypointsGraph
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

from parksim.controller.stanley_controller import StanleyController
from parksim.controller_types import StanleyParams
from parksim.spot_nn.spot_nn import SpotNet
from parksim.spot_nn.feature_generator import SpotFeatureGenerator

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
        self.agents_data_path = '/ParkSim/data/agents_data.pickle'

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
        self.feature_generator = SpotFeatureGenerator()
        self.vehicle_ids_entered = []

        self.queue_length = 0

        # uncomment this to regenerate data used to generate features (1/3)
        # self.discretization = {}
        # self.traj_len = {}
        # self.last_pt = {}
        
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
            inst_centric_generator=None, 
            intent_predictor=None
            )

        vehicle.load_parking_spaces(spots_data_path=spots_data_path)
        vehicle.load_graph(waypoints_graph_path=waypoints_graph_path)
        vehicle.load_maneuver(offline_maneuver_path=offline_maneuver_path)

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
        # TODO: this is a change in spawning
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
                    elif ("ev_charging" not in self.agents_dict[agent] and task["name"] == "PARK" and i > 0) \
                        or ("ev_charging" in self.agents_dict[agent] and self.agents_dict[agent]["ev_charging"] and task["name"] == "PARK" and i > 0 and "target_spot_index" not in task):

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
                self.agents_dict[agent]["task_profile"] = new_tp

            self.add_vehicle(vehicle_id=agent)

        for added in vehicles_to_add:
            del self.agents_dict[added]

    def vehicle_entering(self, vehicle_id):
        return "init_coords" in self.agents_dict[vehicle_id] and self.agents_dict[vehicle_id]["init_coords"][1] > self.y_bound_to_resume_spawning

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
                    self.queue_length = sum([self.time > t for t in self.spawn_entering_time_cumsum])
                    if self.keep_spawn_entering:
                        self.try_spawn_entering()
                    self.try_spawn_exiting()
                else:
                    self.queue_length = sum([self.agents_dict[v]["init_time"] < self.time and self.vehicle_entering(v) for v in self.agents_dict])
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

            # ========== For real-time prediction only
            # add vehicle states to history
            # current_frame_states = []
            # for vehicle in self.vehicles:
            #     current_state_dict = vehicle.get_state_dict()
            #     current_frame_states.append(current_state_dict)
            # self.history.append(current_frame_states)
                
            # intent_pred_results = []
            # ===========

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

                if vehicle_id in self.vehicle_ids_entered and vehicle_id not in self.vehicle_features:
                    self.vehicle_features[vehicle_id] = self.feature_generator.generate_features(vehicle.spot_index, [active_vehicles[id] for id in active_vehicles], self.spawn_interval_mean)

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
            
            self.loops += 1
            self.time += self.timer_period

            if self.should_visualize:

                # Visualize
                for vehicle in self.vehicles:

                    if vehicle.is_all_done():
                        fill = (0, 0, 0, 255)
                    elif vehicle.is_braking:
                        fill = (255, 0, 0, 255)
                    elif vehicle.current_task in ["PARK", "UNPARK"]:
                        fill = (255, 128, 0, 255)
                    else:
                        fill = (0, 255, 0, 255)

                    self.vis.draw_vehicle(state=vehicle.state, fill=fill)
                    # self.vis.draw_line(points=np.array([vehicle.x_ref, vehicle.y_ref]).T, color=(39,228,245, 193))
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
"""
Change these parameters to run tests using the neural network
"""
class RuleBasedSimulatorParams():
    def __init__(self):
        self.seed = 0

        self.num_simulations = 1 # number of simulations run (e.g. times started from scratch)
        self.use_existing_agents = True # replay video data

        # use existing agents
        self.use_existing_entrances = False # have vehicles park in spots that they parked in real life

        # don't use existing agents
        self.spawn_entering_fn = lambda: 30 
        self.spawn_exiting_fn = lambda: 0
        self.spawn_interval_mean_fn = lambda: 2 # (s)

        self.use_existing_obstacles = False # able to park in "occupied" spots from dataset? False if yes, True if no

        self.load_existing_net = True # generate a new net form scratch (and overwrite model.pickle) or use the one stored at self.spot_model_path
        self.use_nn = True # pick spots using NN or not (irrelevant if self.use_existing_entrances is True)
        self.train_nn = False # train NN or not
        self.should_visualize = True # display simulator or not
        self.spot_model_path = '/Parksim/python/parksim/spot_nn/model.pickle' # this model is trained with loss [sum([(net_discount ** i) * simulator.vehicle_non_idle_times[v + i] for i in range(5)]
        self.losses_csv_path = '/parksim/python/parksim/spot_nn/losses.csv' # where losses are stored

        # load net
        if self.load_existing_net:
            self.net = torch.load(str(Path.home()) + self.spot_model_path)
        else:
            self.net = SpotNet()

        self.feature_generator = SpotFeatureGenerator()

    # run simulations, including training the net (if necessary) and saving/printing any results
    def run_simulations(self, ds, vis):
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
            total_loss = self.update_net(simulator)
            losses.append(total_loss / len(simulator.vehicle_ids_entered) if total_loss is not None else 0)
            average_times.append(sum([simulator.vehicle_non_idle_times[i] for i in simulator.vehicle_ids_entered]) / len(simulator.vehicle_ids_entered))
            print('Results: ' + (str(losses[-1]) if total_loss is not None else 'N/A') + ' average loss, ' + str(average_times[-1]) + ' average entering time')

        
        with open(str(Path.home()) + self.losses_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Simulation", "Loss", "Average Entering Time"])
            for i in range(len(losses)):
                writer.writerow([i, losses[i], average_times[i]])

    # spot selection algorithm
    def choose_spot(self, simulator: RuleBasedSimulator, empty_spots: List[int], active_vehicles: List[RuleBasedStanleyVehicle]):
        if self.use_nn:
            return min([spot for spot in empty_spots], key=lambda spot: self.net(self.feature_generator.generate_features(spot, active_vehicles, simulator.spawn_interval_mean)))
            # if self.num_vehicles % 2 == 0:
            #     chosen_spot = min([spot for spot in empty_spots], key=lambda spot: self.spot_net(SpotFeatureGenerator.generate_features(self.add_vehicle(spot_index=spot, for_nn=True), active_vehicles, self.spawn_interval_mean)))
            # else:
            #     chosen_spot = min([spot for spot in empty_spots], key=lambda spot: self.vanilla_net(SpotFeatureGenerator.generate_features(self.add_vehicle(spot_index=spot, for_nn=True), active_vehicles, self.spawn_interval_mean)))
        else:
            # return np.random.choice(empty_spots)
            # return np.random.choice([i for i in empty_spots if (i >= 46 and i <= 66) or (i >= 70 and i <= 91) or (i >= 137 and i <= 158) or (i >= 162 and i <= 183)])
            return min([spot for spot in empty_spots], key=lambda spot: np.linalg.norm([simulator.entrance_coords[0] - simulator.parking_spaces[spot][0], simulator.entrance_coords[1] - simulator.parking_spaces[spot][1]]))

    # target function for neural net
    def target(self, simulator: RuleBasedSimulator, vehicle_id: int, vehicles_included: List[int]):
        net_discount = 0.9
        # return torch.FloatTensor([sum([(net_discount ** i) * simulator.vehicle_non_idle_times[vehicles_included[i]] for i in range(len(vehicles_included))])])
        return torch.FloatTensor([simulator.vehicle_non_idle_times[vehicle_id]])

    # update network and return loss
    def update_net(self, simulator: RuleBasedSimulator):
        num_vehicles_included = 3
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
    ds.load(home_path + '/dlp-dataset/data/DJI_0022')
    print("Dataset loaded.")

    vis = RealtimeVisualizer(ds, VehicleBody())

    params = RuleBasedSimulatorParams()

    if params.seed is not None:
        np.random.seed(params.seed)
        
    params.run_simulations(ds, vis)
    

if __name__ == "__main__":
    main()