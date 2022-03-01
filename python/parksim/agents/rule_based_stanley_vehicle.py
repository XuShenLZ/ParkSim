from typing import Dict, List, Set, Tuple
import numpy as np
from pathlib import Path
import pickle
import time
import array

from parksim.path_planner.offline_maneuver import OfflineManeuver

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController

from parksim.pytypes import VehiclePrediction, VehicleState
from parksim.route_planner.a_star import AStarGraph, AStarPlanner
from parksim.route_planner.graph import Vertex, WaypointsGraph
from parksim.utils.get_corners import get_vehicle_corners
from parksim.utils.interpolation import interpolate_states_inputs
from parksim.vehicle_types import VehicleBody, VehicleConfig, VehicleInfo

class RuleBasedStanleyVehicle(AbstractAgent):
    def __init__(self, vehicle_id: int, vehicle_body: VehicleBody, vehicle_config: VehicleConfig, controller: StanleyController = StanleyController(), motion_predictor: StanleyController = StanleyController(), inst_centric_generator = None, intent_predictor = None):
        self.vehicle_id = vehicle_id

        # State and Reference Waypoints
        self.state: VehicleState = VehicleState() # state
        self.info: VehicleInfo = VehicleInfo() # Info
        self.disp_text: str = str(self.vehicle_id)
        
        self.state_hist: List[VehicleState] = [] # State history

        self.x_ref = [] # x coordinates for waypoints
        self.y_ref = [] # y coordinates for waypoints
        self.yaw_ref = [] # yaws for waypoints
        self.v_ref = 0 # target speed

        # Dimensions
        self.vehicle_body = vehicle_body

        self.vehicle_config = vehicle_config

        # Controller and predictor
        self.controller = controller
        self.motion_predictor = motion_predictor
        self.intent_predictor = intent_predictor # cnnV2 
        self.inst_centric_generator = inst_centric_generator

        self.target_idx = 0
        self.reached_tgt = False
        
        # parking stuff
        self.graph: WaypointsGraph = None
        self.entrance_vertex: int = None

        self.occupancy = None
        self.parking_spaces = None
        self.north_spot_idx_ranges: List[Tuple[int, int]] = None
        self.spot_y_offset: float = None

        self.spot_index = None
        self.should_overshoot = False # overshooting or undershooting the spot?
        self.park_start_coords = None

        self.offline_maneuver: OfflineManeuver = None
        self.overshoot_ranges: Dict[str, List[Tuple[int]]] = None

        self.parking_flag = "" # "", "PARKING", "UNPARKING"
        self.parking_start_time = float('inf') # inf means haven't start parking or unparking. Anything above 0 is parking

        self.parking_maneuver = None
        self.parking_step = 0
        
        # unparking stuff
        self.unparking_maneuver = None
        self.unparking_step = -1
        
        # braking stuff
        self.is_braking = False # are we braking?
        self._pre_brake_target_speed = 0 # speed to restore when unbraking
        self.priority = 0 # priority for going after braking
        self.waiting_for: int = 0 # vehicle waiting for before we go. We start indexing vehicles from 1, so 0 means no vehicle
        self.waiting_for_unparker = False # need special handling for waiting for unparker

        # ============= Information of other vehicles ===========
        self.other_vehicles: Set(int) = set() # Other vehicle ids
        self.nearby_vehicles: Set(int) = set() # Nearby vehicles that we are interested
        self.other_state: Dict[int, VehicleState] = {}
        self.other_ref_pose: Dict[int, VehiclePrediction] = {}
        self.other_ref_v: Dict[int, float] = {}
        self.other_target_idx: Dict[int, int] = {}
        self.other_priority: Dict[int, int] = {}
        self.other_parking_flag: Dict[int, str] = {} # The parking flag of other vehicle
        self.other_parking_progress: Dict[int, str] = {} # Other vehicles will broadcast "PARKING" if vehicle.is_parking(), "UNPARKING" if vehicle.is_unparking(), None otherwise
        self.other_parking_start_time: Dict[int, float] = {}
        self.other_is_braking: Dict[int, str] = {}
        self.other_waiting_for: Dict[int, int] = {}

        # ============== Method to exchange information
        self.method_to_change_central_occupancy = None

    def set_ref_pose(self, x_ref: List[float], y_ref: List[float], yaw_ref: List[float]):
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.yaw_ref = yaw_ref

        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        self.target_idx = self.controller.calc_target_index(self.state)[0] # waypoint the vehicle is targeting

    def set_ref_v(self, v_ref: float):
        self.v_ref = v_ref

    def set_target_idx(self, target_idx: int):
        self.target_idx = target_idx

    def set_spot_idx(self, spot_index: int):
        self.spot_index = spot_index
        if self.spot_index < 0:
            self.parking_flag = "UNPARKING"

    def load_parking_spaces(self, spots_data_path: str):
        home_path = str(Path.home())
        with open(home_path + spots_data_path, 'rb') as f:
            data = pickle.load(f)
            self.parking_spaces = data['parking_spaces']
            self.overshoot_ranges = data['overshoot_ranges']
            self.north_spot_idx_ranges = data['north_spot_idx_ranges']
            self.spot_y_offset = data['spot_y_offset']

    def load_graph(self, waypoints_graph_path: str):
        """
        waypoints_graph_path: path to WaypointGraph object pickle
        entrance_coords: The (x,y) coordinates of the entrance
        """
        home_path = str(Path.home())
        with open(home_path + waypoints_graph_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            entrance_coords = data['entrance_coords']

        self.entrance_vertex = self.graph.search(entrance_coords)

    def load_maneuver(self, offline_maneuver_path: str):
        home_path = str(Path.home())
        self.offline_maneuver = OfflineManeuver(pickle_file=home_path+offline_maneuver_path)

    def load_intent_model(self, model_path: str):
        """
        load_graph must be called before load_intent_model.
        """
        home_path = str(Path.home())
        self.intent_predictor.load_model(waypoints=self.graph, model_path=home_path + model_path)
    
    def compute_ref_path(self, graph_sol: AStarGraph, offset: float = None):
        if not offset:
            offset = self.vehicle_config.offset

        if self.spot_index < 0:
            # exiting
            x_ref, y_ref, yaw_ref = graph_sol.compute_ref_path(offset)
        else:
            # parking
            last_edge = graph_sol.edges[-1]
            pointed_right = last_edge.v2.coords[0] - last_edge.v1.coords[0] > 0

            if pointed_right:
                overshoot_ranges = self.overshoot_ranges['pointed_right']
            else:
                overshoot_ranges = self.overshoot_ranges['pointed_left']

            self.should_overshoot = any([self.spot_index >= r[0] and self.spot_index <= r[1] for r in overshoot_ranges])

            last_x, last_y = last_edge.v2.coords

            if self.should_overshoot:
                # add point past the final waypoint, that signifies going past the spot by 4 meters, so it parks in the right place
                # if the last edge was pointed right, offset to the right
                if pointed_right:
                    new_vertex = Vertex(np.array([last_x+4, last_y]))
                else:
                    new_vertex = Vertex(np.array([last_x-4, last_y]))
                graph_sol.vertices.append(new_vertex)
            else:
                # if the last edge was pointed right, offset to the left
                if pointed_right:
                    new_vertex = Vertex(np.array([last_x-4, last_y]))
                else:
                    new_vertex = Vertex(np.array([last_x+4, last_y]))

                last_waypoint = None
                for i, v in enumerate(reversed(graph_sol.vertices)):
                    if pointed_right:
                        if v.coords[0] < new_vertex.coords[0]:
                            last_waypoint = -i-1
                            break
                    else:
                        if v.coords[0] > new_vertex.coords[0]:
                            last_waypoint = -i-1
                            break

                graph_sol.vertices = graph_sol.vertices[:last_waypoint+1]
                graph_sol.vertices.append(new_vertex)

            x_ref, y_ref, yaw_ref = graph_sol.compute_ref_path(offset)

        return x_ref, y_ref, yaw_ref

    def start_vehicle(self):

        assert self.parking_spaces is not None, "Please run load_parking_spaces first."
        assert self.spot_index is not None, "Please run set_spot_idx first."
        assert self.graph is not None, "Please run load_graph first."

        is_north_spot = any([abs(self.spot_index) >= r[0] and abs(self.spot_index) <= r[1] for r in self.north_spot_idx_ranges])
        y_offset = -self.spot_y_offset if is_north_spot else self.spot_y_offset
        waypoint_coords = [self.parking_spaces[abs(self.spot_index)][0], self.parking_spaces[abs(self.spot_index)][1] + y_offset]

        if self.spot_index > 0: # entering
            graph_sol = AStarPlanner(self.graph.vertices[self.entrance_vertex], self.graph.vertices[self.graph.search(waypoint_coords)]).solve()
        else: # exiting
            graph_sol = AStarPlanner(self.graph.vertices[self.graph.search(waypoint_coords)], self.graph.vertices[self.entrance_vertex]).solve()

        x_ref, y_ref, yaw_ref = self.compute_ref_path(graph_sol=graph_sol)

        # Set initial state
        if self.spot_index > 0: # entering
            self.state.x.x = x_ref[0]
            self.state.x.y = y_ref[0]
            self.state.e.psi = yaw_ref[0]
        else: # start parked
            # randomize if pointing up or down to start
            self.state.x.x = self.parking_spaces[-self.spot_index][0]
            self.state.x.y = self.parking_spaces[-self.spot_index][1]
            self.state.e.psi = np.pi / 2 if np.random.rand() < 0.5 else -np.pi / 2

        self.set_ref_pose(x_ref, y_ref, yaw_ref)
        self.set_ref_v(0)
    
    def reached_target(self):
        if not self.reached_tgt:
            dist = np.linalg.norm([self.state.x.x - self.x_ref[-1], self.state.x.y - self.y_ref[-1]])
            ang = ((np.arctan2(self.y_ref[-1] - self.state.x.y, self.x_ref[-1] - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
            self.reached_tgt = dist < self.vehicle_config.braking_distance/2 and ang > (np.pi / 2) and ang < (3 * np.pi / 2)
        return self.reached_tgt

    def num_waypoints(self):
        return len(self.x_ref)

    def set_method_to_change_central_occupancy(self, method):
        self.method_to_change_central_occupancy = method

    def get_central_occupancy(self, occupancy):
        """
        Get the parking occupancy
        """
        self.occupancy = occupancy

    def change_central_occupancy(self, idx, new_value):
        """
        Request to change the occupancy
        """
        method = self.method_to_change_central_occupancy
        if callable(method):
            # Call ROS service to change occupancy
            method(idx, new_value)
        else:
            method[idx] = new_value

    def get_info(self):
        self.info.ref_pose.x = array.array('d', self.x_ref)
        self.info.ref_pose.y = array.array('d', self.y_ref)
        self.info.ref_pose.psi = array.array('d', self.yaw_ref)
        self.info.ref_v = self.v_ref
        self.info.target_idx = self.target_idx
        self.info.priority = self.priority
        self.info.parking_flag = self.parking_flag

        if self.is_parking():
            self.info.parking_progress = "PARKING"
        elif self.is_unparking():
            self.info.parking_progress = "UNPARKING"
        else:
            self.info.parking_progress = ""

        self.info.is_braking = self.is_braking
        self.info.parking_start_time = self.parking_start_time
        self.info.waiting_for = self.waiting_for

        self.info.disp_text = self.disp_text

        return self.info

    def get_other_info(self, active_vehicles: Dict[int, AbstractAgent]):
        """
        The equavilence of ROS subscribers
        """
        active_ids = set([id for id in active_vehicles if id != self.vehicle_id])
        
        self.other_vehicles.update(active_ids)
        ids_to_delete = self.other_vehicles - active_ids
        self.other_vehicles.difference_update(ids_to_delete)

        for id in active_vehicles:
            if id == self.vehicle_id:
                continue
            
            v = active_vehicles[id]

            self.other_state[id] = v.state
            ref_pose = VehiclePrediction()
            ref_pose.x = v.x_ref
            ref_pose.y = v.y_ref
            ref_pose.psi = v.yaw_ref
            self.other_ref_pose[id] = ref_pose
            self.other_ref_v[id] = v.v_ref
            self.other_target_idx[id] = v.target_idx
            self.other_priority[id] = v.priority

            # NOTE: distinguish between "mid_park" and "parking"
            self.other_parking_flag[id] = v.parking_flag
            if v.is_parking():
                self.other_parking_progress[id] = "PARKING"
            elif v.is_unparking():
                self.other_parking_progress[id] = "UNPARKING"
            else:
                self.other_parking_progress[id] = ""

            self.other_is_braking[id] = v.is_braking
            self.other_parking_start_time[id] = v.parking_start_time
            self.other_waiting_for[id] = v.waiting_for

    def dist_from(self, other_id: int):
        """
        Compute Euclidean distance with the other vehicle
        """
        return np.linalg.norm([self.other_state[other_id].x.x - self.state.x.x, self.other_state[other_id].x.y - self.state.x.y])

    def will_crash_with(self) -> Set[int]:
        will_crash_with = set()

        # create states for looking ahead
        look_ahead_state = self.state.copy()
        other_look_ahead_states = [self.other_state[id].copy() for id in self.nearby_vehicles]

        # for each time step, looking ahead
        for _ in range(self.vehicle_config.look_ahead_timesteps):
            # calculate new positions
            self.motion_predictor.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
            self.motion_predictor.set_ref_v(self.v_ref)
            self.motion_predictor.set_target_idx(self.target_idx)
            ai, di, _ = self.motion_predictor.solve(look_ahead_state, self.is_braking)
            self.motion_predictor.step(look_ahead_state, ai, di)

            for id, other_look_ahead_state in zip(self.nearby_vehicles, other_look_ahead_states):
                if id not in will_crash_with: # for efficiency
                    self.motion_predictor.set_ref_pose(self.other_ref_pose[id].x, self.other_ref_pose[id].y, self.other_ref_pose[id].psi)
                    self.motion_predictor.set_ref_v(self.other_ref_v[id])
                    self.motion_predictor.set_target_idx(self.other_target_idx[id])
                    ai, di, _ = self.motion_predictor.solve(other_look_ahead_state, self.other_is_braking[id])
                    self.motion_predictor.step(other_look_ahead_state, ai, di)


            # detect crash
            for id, other_look_ahead_state in zip(self.nearby_vehicles, other_look_ahead_states):
                if id not in will_crash_with: # for efficiency
                    if self.will_collide(look_ahead_state, other_look_ahead_state, self.vehicle_body):
                        # NOTE: Here we assume all other vehicles have the same vehicle body as us
                        will_crash_with.add(id)

        return will_crash_with

    def should_go_before(self, other_id):
        """
        Determines if one car should go before another. Does it based on angles: if one vehicle has gone more past the other vehicle than the other, it should go first.
        """
        this_ang = ((np.arctan2(self.other_state[other_id].x.y - self.state.x.y, self.other_state[other_id].x.x - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
        other_ang = ((np.arctan2(self.state.x.y - self.other_state[other_id].x.y, self.state.x.x - self.other_state[other_id].x.x) - self.other_state[other_id].e.psi) + (2*np.pi)) % (2*np.pi)
        this_ang_centered = this_ang if this_ang < np.pi else this_ang - 2 * np.pi
        other_ang_centered = other_ang if other_ang < np.pi else other_ang - 2 * np.pi
        return abs(this_ang_centered) > abs(other_ang_centered)

    def has_passed(self, this_id: int=None, other_id: int=None, parking_dist_away=None):
        """
        If the rear corners of this vehicle have passed the front corners of the other vehicle, we say this vehicle has passed the other vehicle.
        """
        if this_id is None or this_id == self.vehicle_id:
            this_corners = self.get_corners()
            this_state = self.state
            this_psi = this_state.e.psi
        else:
            this_corners = self.get_corners(self.other_state[this_id])
            this_state = self.other_state[this_id]
            this_psi = this_state.e.psi
        
        if other_id is None or other_id == self.vehicle_id:
            other_corners = self.get_corners()
            other_state = self.state
        else:
            other_corners = self.get_corners(self.other_state[other_id]) # NOTE: For now, assume the other vehicle has the same vehicle body
            other_state = self.other_state[other_id]

        for this_corner in [this_corners[0], this_corners[1]]:
            for other_corner in [other_corners[2], other_corners[3]]:
                ang = ((np.arctan2(other_corner[1] - this_corner[1], other_corner[0] - this_corner[0]) - this_psi) + (2*np.pi)) % (2*np.pi)
                if ang < (np.pi/2) or ang > (3*np.pi)/2:
                    return False
        if parking_dist_away is not None:
            if this_psi > np.pi / 2 and this_psi < np.pi * 3 / 2: # facing west
                if this_state.x.x - other_state.x.x > -parking_dist_away:
                    return False
            else:
                if this_state.x.x - other_state.x.x < parking_dist_away:
                    return False
        return True

    def other_within_parking_box(self, other_id):
        ang = ((np.arctan2(self.other_state[other_id].x.y - self.state.x.y, self.other_state[other_id].x.x - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
        dist = self.dist_from(other_id)
        if ang < self.vehicle_config.parking_ahead_angle or ang > 2 * np.pi - self.vehicle_config.parking_ahead_angle:
            return dist < 2*self.vehicle_config.parking_radius
        else:
            return dist < self.vehicle_config.parking_radius

    def update_state(self):
        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        self.controller.set_ref_v(self.v_ref)
        self.controller.set_target_idx(self.target_idx)
        # get acceleration toward target speed (ai), amount we should turn (di), and next target (target_idx)
        ai, di, self.target_idx = self.controller.solve(self.state, self.is_braking)
        # advance state of vehicle (updates x, y, yaw, velocity)
        self.controller.step(self.state, ai, di)
            
    def update_state_parking(self, advance=True):
        if self.parking_maneuver is None: # start parking
            # get parking parameters
            direction = 'west' if self.state.e.psi > np.pi / 2 or self.state.e.psi < -np.pi / 2 else 'east'
            if self.should_overshoot:
                location = 'right' if (direction == 'east') else 'left' # we are designed to overshoot the spot
            else:
                location = 'left' if (direction == 'east') else 'right' # we are designed to undershoot the spot
            pointing = 'up' if np.random.rand() < 0.5 else 'down' # random for diversity
            spot = 'north' if any([self.spot_index >= r[0] and self.spot_index <= r[1] for r in self.north_spot_idx_ranges]) else 'south'
            
            # get parking maneuver
            offline_maneuver = self.offline_maneuver.get_maneuver([self.park_start_coords[0] - 4 if location == 'right' else self.park_start_coords[0] + 4, self.park_start_coords[1]], direction, location, spot, pointing)

            time_seq = np.arange(start=offline_maneuver.t[0], stop=offline_maneuver.t[-1], step=self.controller.dt)
            
            self.parking_maneuver = interpolate_states_inputs(offline_maneuver, time_seq)

            self.parking_start_time = time.time()
            
            
        # idle when done
        step = min(self.parking_step, len(self.parking_maneuver.x) - 1)
        # set state
        self.state.x.x = self.parking_maneuver.x[step]
        self.state.x.y = self.parking_maneuver.y[step]
        self.state.e.psi = self.parking_maneuver.psi[step]
        self.state.v.v = self.parking_maneuver.v[step]

        self.state.u.u_a = self.parking_maneuver.u_a[step]
        self.state.u.u_steer = self.parking_maneuver.u_steer[step]
        
        # update parking step if advancing
        self.parking_step += 1 if advance else 0
        
    def update_state_unparking(self, advance=True):
        if self.unparking_maneuver is None: # start unparking
            # get unparking parameters
            direction = 'west' if self.x_ref[0] > self.x_ref[1] else 'east' # if first direction of travel is left, face west
            location = 'right' if np.random.rand() < 0.5 else 'left' # random for diversity
            pointing = 'up' if self.state.e.psi > 0 else 'down' # determine from state
            spot = 'north' if any([-self.spot_index >= r[0] and -self.spot_index <= r[1] for r in self.north_spot_idx_ranges]) else 'south'
            
            # get parking maneuver
            offline_maneuver = self.offline_maneuver.get_maneuver([self.state.x.x if location == 'right' else self.state.x.x, self.state.x.y - 6.25 if spot == 'north' else self.state.x.y + 6.25], direction, location, spot, pointing)

            time_seq = np.arange(start=offline_maneuver.t[0], stop=offline_maneuver.t[-1], step=self.controller.dt)
            
            self.unparking_maneuver = interpolate_states_inputs(offline_maneuver, time_seq)
            
            # set initial unparking state
            self.unparking_step = len(self.unparking_maneuver.x) - 1

            self.parking_start_time = time.time()
            
        # get step
        step = self.unparking_step
            
        # set state
        self.state.x.x = self.unparking_maneuver.x[step]
        self.state.x.y = self.unparking_maneuver.y[step]
        self.state.e.psi = self.unparking_maneuver.psi[step]
        self.state.v.v = self.unparking_maneuver.v[step]

        self.state.u.u_a = self.unparking_maneuver.u_a[step]
        self.state.u.u_steer = self.unparking_maneuver.u_steer[step]
        
        if self.unparking_step == 0: # done unparking
            self.parking_flag = ""
            self.change_central_occupancy(-self.spot_index, False)
        else:
            # update parking step if advancing
            self.unparking_step -= 1 if advance else 0

    def get_corners(self, state: VehicleState=None, vehicle_body: VehicleBody=None):
        """
        state, vehicle_body: If computing the state of vehicle itself, leave this optional. If computing another vehicle, fill in the corresponding property
        """
        if state is None:
            state = self.state

        if vehicle_body is None:
            vehicle_body = self.vehicle_body

        return get_vehicle_corners(state=state, vehicle_body=vehicle_body)

    def brake(self):
        """
        Set target speed to 0 and turn on brakes, which make deceleration faster
        """
        self._pre_brake_target_speed = self.v_ref
        self.v_ref = 0
        self.is_braking = True

    def unbrake(self):
        """
        Set target speed back to what it was. Only does something if braking
        """
        if self.is_braking:
            self.v_ref = self._pre_brake_target_speed
            self.is_braking = False
            self.priority = 0
            self.waiting_for = 0
    
    def is_parking(self):
        """
        Are we in the middle of a parking manuever? If this is False, traffic should have the right of way, else this vehicle should have the right of way
        """
        return self.parking_flag == "PARKING" and self.parking_maneuver is not None and self.parking_step > 0 and self.parking_step < len(self.parking_maneuver.x) - 1

    def is_unparking(self):
        return (self.parking_flag == "UNPARKING" and self.unparking_maneuver is not None and self.unparking_step < len(self.unparking_maneuver.x) - 1 and self.unparking_step > 0) and not (self.parking_flag == "UNPARKING" and self.unparking_step == -1)
    
    def is_all_done(self):
        """
        Have we finished the parking maneuver or we have reached the exit?
        """
        
        return (self.parking_flag == "PARKING" and self.parking_maneuver is not None and self.parking_step >= len(self.parking_maneuver.x)) or (self.spot_index < 0 and self.reached_tgt)
    
    def update_nearby_vehicles(self, radius=None):
        """
        radius: (Optional) vehicles inside this radius are considered as "nearby". If left empty, we will use vehicle_config.crash_check_radius
        """
        if not radius:
            radius = self.vehicle_config.crash_check_radius

        self.nearby_vehicles.clear()

        for id in self.other_vehicles:
            dist = self.dist_from(id)

            if dist < radius:
                self.nearby_vehicles.add(id)
            

    def solve(self):
        """
        Having other_vehicle_objects here is just to mimic the ROS service to change values of the other vehicle. Should use this to acquire information
        """
        # Firstly update the set of nearby_vehicles
        self.update_nearby_vehicles()

        # driving control

        if self.parking_flag:
            pass
        elif not self.reached_target():
            # normal driving (haven't reached pre-parking point)

            # braking controller
            if not self.is_braking:
                # normal speed controller if not braking
                if self.target_idx < self.num_waypoints() - self.vehicle_config.steps_to_end:
                    self.set_ref_v(self.vehicle_config.v_max)
                else:
                    self.set_ref_v(self.vehicle_config.v_end)

                # detect parking and unparking
                nearby_parkers = [id for id in self.nearby_vehicles if self.other_parking_progress[id] and self.other_within_parking_box(id)]

                if nearby_parkers:
                    # should only be one nearby parker, since they wait for each other
                    parker_id = nearby_parkers[0]
                    self.brake()
                    self.waiting_for = parker_id
                    self.priority = -1

                    if self.other_parking_flag[parker_id] == "UNPARKING":
                        self.waiting_for_unparker = True

                else: # No one is parking
                    ids_will_crash_with = list(self.will_crash_with())

                    # If will crash
                    if ids_will_crash_with:
                        # variable to tell where to go next (default is braking)
                        going_to_brake = True

                        # set priority
                        other_id = ids_will_crash_with[0] # TODO: what if there are multiple cars it will crash with?

                        if not any([self.should_go_before(id) for id in ids_will_crash_with]):
                            going_to_brake = True # go straight to waiting, no priority calculations necessary

                            self.priority = self.other_priority[other_id] - 1

                            self.waiting_for = other_id
                        else: # leading car
                            going_to_brake = False # don't brake
                        
                        if going_to_brake:
                            self.brake()

            else: # waiting / braking
                # parking
                if self.waiting_for != 0 and self.other_parking_flag[self.waiting_for] == "PARKING":
                    if self.waiting_for not in self.nearby_vehicles:
                        self.unbrake()
                        
                elif self.waiting_for != 0 and self.waiting_for_unparker:
                    if self.other_parking_flag[self.waiting_for] != "UNPARKING":
                        self.waiting_for_unparker = False
                        self.unbrake()

                else:
                    # other (standard) cases

                    should_unbrake = False
                    # go if going first
                    if self.waiting_for == 0:
                        should_unbrake = True
                    else:                        
                        if (self.waiting_for not in self.nearby_vehicles  
                            or (not self.other_is_braking[self.waiting_for] 
                            and self.has_passed(this_id=self.waiting_for))
                            or self.dist_from(self.waiting_for) > self.vehicle_config.braking_distance):
                            should_unbrake = True
                        elif self.other_waiting_for[self.waiting_for] == self.vehicle_id: # if the vehicle you're waiting for is waiting for you
                            # you should go
                            # TODO: this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                            # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                            should_unbrake = True

                    if should_unbrake:
                        self.unbrake() # also sets brake_state to NOT_BRAKING

        else:
            # if reached target (pre-parking point), start parking
            self.set_ref_v(0)
            if self.spot_index > 0:
                self.parking_flag = "PARKING"
            self.priority = 1 # high priority for parkers

        if self.parking_flag == "PARKING":
            # wait for coast to be clear, then start parking
            # everyone within range should be braking or parking or unparking

            should_go = (self.parking_maneuver is not None and self.parking_step > 0) \
                or all([
                    self.other_parking_flag[id] == "" and self.has_passed(other_id=id) 
                    or (self.other_parking_flag[id] == "UNPARKING" and self.other_parking_progress[id] == "")
                    or (self.other_parking_flag[id] == "PARKING" and self.other_parking_start_time[id] > self.parking_start_time)
                    or self.dist_from(id) >= 2*self.vehicle_config.parking_radius for id in self.nearby_vehicles
                    ])

            if self.park_start_coords is None:
                self.park_start_coords = (self.state.x.x - self.vehicle_config.offset * np.sin(self.state.e.psi), self.state.x.y + self.vehicle_config.offset * np.cos(self.state.e.psi))
            self.update_state_parking(should_go)
        elif self.parking_flag == "UNPARKING": # wait for coast to be clear, then start unparking
            # to start unparking, everyone within range should be (normal driving and far past us) or (waiting to unpark and spawned after us)
            # always yields to parkers in the area

            should_go = (self.unparking_maneuver is not None and self.unparking_step < len(self.unparking_maneuver.x) - 1) \
                or (all([
                    (self.other_parking_flag[id] == "" and self.has_passed(this_id=id, parking_dist_away=7)) 
                    or (self.other_parking_flag[id] == "UNPARKING" and self.other_parking_start_time[id] > self.parking_start_time)
                    or self.dist_from(id) >= 2*self.vehicle_config.parking_radius for id in self.nearby_vehicles
                    ]) 
                    and all([self.other_parking_progress[id] != "UNPARKING" or np.linalg.norm([self.x_ref[0] - self.other_ref_pose[id].x[0], self.y_ref[0] - self.other_ref_pose[id].y[0]]) > 10 for id in self.other_vehicles]))

            self.update_state_unparking(should_go)
        else: 
            self.update_state()

        self.state_hist.append(self.state.copy())

    def predict_intent(self, vehicle_id=None):
        """
        predict the intent of the specific vehicle
        """
        if vehicle_id is None:
            img = self.inst_centric_generator.inst_centric(self.state, self.other_state)
            return self.intent_predictor.predict(img, np.array([self.state.x.x, self.state.x.y]), self.state.e.psi, self.state.v.v, 1.0)
            # Predicting ourselves
        else:
            raise NotImplementedError("Haven't inplement method for predicting other vehicles")
            # Predicting someone else
