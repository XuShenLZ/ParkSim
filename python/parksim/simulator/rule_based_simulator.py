import time
from typing import Dict, List

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVisualizer

from pathlib import Path

import pickle
import os
import glob

import numpy as np
from scipy.io import savemat
from parksim.pytypes import VehicleState

from parksim.vehicle_types import VehicleBody, VehicleConfig, VehicleTask
from parksim.route_planner.graph import WaypointsGraph
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

from parksim.controller.stanley_controller import StanleyController
from parksim.controller_types import StanleyParams

np.random.seed(39)

# These parameters should all become ROS param for simulator and vehicle
spots_data_path = '/ParkSim/data/spots_data.pickle'
offline_maneuver_path = '/ParkSim/data/parking_maneuvers.pickle'
waypoints_graph_path = '/ParkSim/data/waypoints_graph.pickle'
intent_model_path = '/ParkSim/data/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'

class RuleBasedSimulator(object):
    def __init__(self, dataset: Dataset, vis: RealtimeVisualizer):

        self.timer_period = 0.1

        self.blocked_spots = [42, 43, 44, 45, 64, 65, 66, 67, 68, 69, 92, 93, 94, 110, 111, 112, 113, 114, 115, 134, 135, 136, 156, 157, 158, 159, 160, 161, 184, 185, 186, 202, 203, 204, 205, 206, 207, 226, 227, 228, 248, 249, 250, 251, 252, 253, 276, 277, 278, 294, 295, 256, 297, 298, 299, 318, 319, 320, 340, 341, 342, 343, 344, 345] # Spots to be blocked in advance: 3 left and 3 right spaces of each row, except right spaces of right row, since doesn't unpark into an aisle
        self.entrance_coords = [14.38, 76.21]

        self.spawn_entering = 30
        self.spawn_exiting = 30
        self.y_bound_to_resume_spawning = 70
        self.spawn_interval_mean = 5 # (s)

        self.spots_data_path = '/ParkSim/data/spots_data.pickle'
        self.agents_data_path = '/ParkSim/data/agents_data.pickle'

        self.use_existing_agents = True

        self.write_log = False
        self.log_path = '/ParkSim/vehicle_log'

        self.dlpvis = DlpVisualizer(dataset)
        self.vis = vis
        self.should_visualize = True

        self.graph = WaypointsGraph()
        self.graph.setup_with_vis(self.dlpvis)

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
        self.spawn_exiting_time = list(np.random.exponential(self.spawn_interval_mean, self.spawn_exiting))

        self.last_enter_id = None
        self.last_enter_state = VehicleState()
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

        self.max_simulation_time = 1200

        self.time = 0.0
        self.loops = 0   
        
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

    def _gen_agents(self):
        home_path = str(Path.home())
        with open(home_path + self.agents_data_path, 'rb') as f:
            self.agents_dict = pickle.load(f)

    # goes to an anchor point
    # convention: if entering, spot_index is positive, and if exiting, it's negative
    def add_vehicle(self, spot_index: int=None, vehicle_body: VehicleBody=VehicleBody(), vehicle_config: VehicleConfig=VehicleConfig(), vehicle_id: int=None):
        
        # Start vehicle indexing from 1
        self.num_vehicles += 1
        if vehicle_id is None:
            vehicle_id = self.num_vehicles

        if self.use_existing_agents:
            agents = pickle.load(open(str(Path.home()) + self.agents_data_path, "rb"))
            agent_dict = agents[vehicle_id]

            vehicle_body.w = agent_dict["width"]
            vehicle_body.l = agent_dict["length"]

        controller_params = StanleyParams(dt=self.timer_period)
        controller = StanleyController(control_params=controller_params, vehicle_body=vehicle_body, vehicle_config=vehicle_config)
        motion_predictor = StanleyController(control_params=controller_params, vehicle_body=vehicle_body, vehicle_config=vehicle_config)

        vehicle = RuleBasedStanleyVehicle(
            vehicle_id=vehicle_id, 
            vehicle_body=vehicle_body, 
            vehicle_config=vehicle_config, 
            controller=controller,
            motion_predictor=motion_predictor,
            inst_centric_generator=None, 
            intent_predictor=None
            )

        vehicle.load_parking_spaces(spots_data_path=spots_data_path)
        vehicle.load_graph(waypoints_graph_path=waypoints_graph_path)
        vehicle.load_maneuver(offline_maneuver_path=offline_maneuver_path)
        # vehicle.load_intent_model(model_path=intent_model_path)

        task_profile = []

        if not self.use_existing_agents:
            if spot_index > 0:
                cruise_task = VehicleTask(
                    name="CRUISE", v_cruise=5, target_spot_index=spot_index)
                park_task = VehicleTask(name="PARK", target_spot_index=spot_index)
                task_profile = [cruise_task, park_task]

                state = VehicleState()
                state.x.x = self.entrance_coords[0] - vehicle_config.offset
                state.x.y = self.entrance_coords[1]
                state.e.psi = - np.pi/2

                vehicle.set_vehicle_state(state=state)
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
                    task_profile.append(VehicleTask(name="IDLE", duration=task["duration"]))
                elif task["name"] == "PARK":
                    task_profile.append(VehicleTask(name="PARK", target_spot_index=task["target_spot_index"]))
                elif task["name"] == "UNPARK":
                    task_profile.append(VehicleTask(name="UNPARK", target_spot_index=task["target_spot_index"]))
                elif task["name"] == "CRUISE":
                    if "target_coords" in task:
                        task_profile.append(VehicleTask(name="CRUISE", v_cruise=task["v_cruise"], target_coords=task["target_coords"]))
                    else:
                        task_profile.append(VehicleTask(name="CRUISE", v_cruise=task["v_cruise"], target_spot_index=task["target_spot_index"]))

            if "init_spot" in agent_dict:
                init_spot = agent_dict["init_spot"]
                init_heading = agent_dict["init_heading"]
                vehicle.set_vehicle_state(spot_index=init_spot, heading=init_heading)
            else:
                agent_state = VehicleState()
                agent_state.x.x = agent_dict["init_coords"][0]
                agent_state.x.y = agent_dict["init_coords"][1]
                agent_state.e.psi = agent_dict["init_heading"]
                agent_state.v.v = agent_dict["init_v"]
                vehicle.set_vehicle_state(state=agent_state)

        vehicle.set_task_profile(task_profile=task_profile)
        vehicle.execute_next_task()

        self.vehicle_non_idle_times[vehicle_id] = 0
        self.last_enter_state = vehicle.state

        self.vehicles.append(vehicle)
    
    def try_spawn_entering(self):
        current_time = self.time

        if self.spawn_entering_time and current_time - self.last_enter_time > self.spawn_entering_time[0]:
            empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
            chosen_spot = np.random.choice(empty_spots)
            self.add_vehicle(chosen_spot) # pick from empty spots randomly
            self.occupied[chosen_spot] = True
            self.spawn_entering_time.pop(0)

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
        added_vehicles = []

        for agent in self.agents_dict:
            if self.agents_dict[agent]["init_time"] < current_time:
                self.add_vehicle(vehicle_id=agent)
                added_vehicles.append(agent)

        for added in added_vehicles:
            print(str(added), self.agents_dict[added]["task_profile"])
            del self.agents_dict[added]

    def run(self):

        if self.write_log:
            # write logs
            log_dir_path = str(Path.home()) + self.log_path
            if not os.path.exists(log_dir_path):
                os.mkdir(log_dir_path)

        # while not run out of time and we have not reached the last waypoint yet
        while self.max_simulation_time >= self.time:

            if self.should_visualize:
                if not self.vis.is_running():
                    self.vis.render()
                    continue

                # clear visualizer
                self.vis.clear_frame()

            # If vehicle left entrance area, start spawning another one
            if self.last_enter_state.x.y < self.y_bound_to_resume_spawning:
                self.keep_spawn_entering = True

            if self.sim_is_running:
                if not self.use_existing_agents:
                    if self.keep_spawn_entering:
                        self.try_spawn_entering()
                    self.try_spawn_exiting()
                else:
                    self.try_spawn_existing()

            active_vehicles: Dict[int, RuleBasedStanleyVehicle] = {}
            for vehicle in self.vehicles:
                if not vehicle.is_all_done():
                    active_vehicles[vehicle.vehicle_id] = vehicle

            if not self.spawn_entering_time and not self.spawn_exiting_time and not active_vehicles:
                print("No Active Vehicles")
                break

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
                    self.vehicle_non_idle_times[vehicle.vehicle_id] += self.timer_period

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

            if self.loops % 100 == 0:
                print(self.time)

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

def main():
    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    print('Loading dataset...')
    ds.load(home_path + '/dlp-dataset/data/DJI_0022')
    print("Dataset loaded.")

    vis = RealtimeVisualizer(ds, VehicleBody())

    simulator = RuleBasedSimulator(dataset=ds, vis=vis)

    simulator.run()



if __name__ == "__main__":
    main()