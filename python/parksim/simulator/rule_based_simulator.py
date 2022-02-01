from typing import Dict, List

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVisualizer

from pathlib import Path

import numpy as np

from parksim.vehicle_types import VehicleBody, VehicleConfig
from parksim.route_planner.graph import WaypointsGraph
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

np.random.seed(654) # ones with interesting cases: 20, 33, 44, 60
# 44, 654 has a unparking car ignoring braked vehicles

# These parameters should all become ROS param for simulator and vehicle
parking_spaces_path = '/ParkSim/parking_spaces.npy'
offline_maneuver_path = '/ParkSim/parking_maneuvers.pickle'
waypoints_graph_path = '/ParkSim/waypoints_graph.pickle'
# Add the model path here
# intent_model_path = ''
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

        # anchor spots
        self.anchor_points = anchor_points

        # spawn stuff
        self.spawn_wait = 50 # number of timesteps between cars spawning

        self.spawn_entering = 3 # number of vehicles to enter
        self.spawn_exiting = 3 # number of vehicles to exit
        self.spawn_exiting_loops = np.random.choice(range(self.spawn_exiting * self.spawn_wait), self.spawn_exiting)

        self.num_vehicles = 0
        self.vehicles: List[RuleBasedStanleyVehicle] = []

        self.max_simulation_time = 150

        self.time = 0.0
        self.loops = 0

        # crash detection
        self.did_crash = False
        self.crash_polytopes = None

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
        vehicle.load_parking_spaces(parking_spaces_path=parking_spaces_path, north_spot_idx_ranges=north_spot_idx_ranges, spot_y_offset=spot_y_offset)
        vehicle.set_anchor(going_to_anchor=spot_index>0, spot_index=spot_index, should_overshoot=False, anchor_points=anchor_points, anchor_spots=anchor_spots)
        vehicle.load_graph(waypoints_graph_path=waypoints_graph_path, entrance_coords=entrance_coords)
        vehicle.load_maneuver(offline_maneuver_path=offline_maneuver_path, overshoot_ranges=overshoot_ranges)
        # vehicle.load_intent_model(path=pth)
        vehicle.start_vehicle()

        self.num_vehicles += 1
        self.vehicles.append(vehicle)
    

    def run(self):
        # while not run out of time and we have not reached the last waypoint yet
        while self.max_simulation_time >= self.time:

            # clear visualizer
            self.vis.clear_frame()
            
            # spawn vehicles
            if self.loops % self.spawn_wait == 0 and self.spawn_entering > 0: # entering
                self.add_vehicle(np.random.choice(self.anchor_points)) # pick from the anchor points at random
                self.spawn_entering -= 1
            if self.loops in self.spawn_exiting_loops: # spawn in random empty spot
                empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
                chosen_spot = np.random.choice(empty_spots)
                self.add_vehicle(-1 * chosen_spot)
                self.occupied[chosen_spot] = True

            active_vehicles: Dict[int, RuleBasedStanleyVehicle] = {}
            for vehicle in self.vehicles:
                if not vehicle.is_all_done():
                    active_vehicles[vehicle.vehicle_id] = vehicle

            if not active_vehicles:
                print("No Active Vehicles")
                break
                
            for vehicle_id in active_vehicles:
                vehicle = active_vehicles[vehicle_id]

                vehicle.get_other_info(active_vehicles)
                vehicle.set_method_to_get_central_occupancy(self.occupied)
                vehicle.set_method_to_change_central_occupancy(self.occupied)

                vehicle.solve()
                # result = vehicle.predict_intent()
                
            self.loops += 1
            self.time += 0.1

            # Visualize
            for vehicle in self.vehicles:

                if vehicle.is_all_done():
                    fill = (0, 0, 0, 255)
                elif vehicle.is_braking():
                    fill = (255, 0, 0, 255)
                elif vehicle.parking_flag:
                    fill = (255, 128, 0, 255)
                else:
                    fill = (0, 255, 0, 255)

                self.vis.draw_vehicle(state=vehicle.state, fill=fill)
                self.vis.draw_line(points=np.array([vehicle.x_ref, vehicle.y_ref]).T, color=(39,228,245, 193))
                on_vehicle_text = "N" if vehicle.priority is None else round(vehicle.priority, 3)
                self.vis.draw_text([vehicle.state.x.x - 2, vehicle.state.x.y + 2], on_vehicle_text, size=25)
                # self.vis.draw_text([x,y], prob, size, color)
            self.vis.render()

def main():
    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    print('Loading dataset...')
    ds.load(home_path + '/dlp-dataset/data/DJI_0012')
    print("Dataset loaded.")

    vis = RealtimeVisualizer(ds, VehicleBody())

    simulator = RuleBasedSimulator(dataset=ds, vis=vis, anchor_points=anchor_points)

    simulator.run()



if __name__ == "__main__":
    main()