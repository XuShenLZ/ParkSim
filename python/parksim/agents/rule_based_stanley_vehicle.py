from typing import Dict, List, Set, Tuple
import numpy as np
from pathlib import Path
import pickle

from parksim.path_planner.offline_maneuver import OfflineManeuver

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController

from parksim.pytypes import VehiclePrediction, VehicleState
from parksim.route_planner.a_star import AStarPlanner
from parksim.route_planner.graph import WaypointsGraph
from parksim.utils.spline import calc_spline_course
from parksim.vehicle_types import VehicleBody, VehicleConfig

class RuleBasedStanleyVehicle(AbstractAgent):
    def __init__(self, vehicle_id: int, vehicle_body: VehicleBody, vehicle_config: VehicleConfig, controller: StanleyController = StanleyController(), motion_predictor: StanleyController = StanleyController(), intent_predictor = None):
        self.vehicle_id = vehicle_id

        # State and Reference Waypoints
        self.state: VehicleState = VehicleState() # state

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
        self.intent_predictor = intent_predictor

        self.target_idx = 0
        self.reached_tgt = False
        
        # parking stuff
        self.graph: WaypointsGraph = None
        self.entrance_vertex: int = None

        self.occupancy = None
        self.parking_spaces = None

        self.anchor_points = []
        self.anchor_spots = []

        self.going_to_anchor = True # going to anchor if parking, not if exiting
        self.spot_index = None
        self.should_overshoot = False # overshooting or undershooting the spot?
        self.park_start_coords = None

        self.offline_maneuver: OfflineManeuver = None
        self.overshoot_ranges: Dict[str, List[Tuple[int]]] = None

        self.parking_flag = None # None, "PARKING", "UNPARKING"

        self.parking_maneuver_state = None
        self.parking_step = 0
        
        # unparking stuff
        self.unparking_maneuver_state = None
        self.unparking_step = 0
        
        # braking stuff
        self.braking_flag = "NOT_BRAKING" # are we braking? "NOT_BRAKING", "BRAKING", "WAITING"
        self._pre_brake_target_speed = 0 # speed to restore when unbraking
        self.priority = None # priority for going after braking
        self.crash_set: Set[RuleBasedStanleyVehicle] = set() # vehicles that we will crash with
        self.waiting_for: int = None # vehicle waiting for before we go
        self.waiting_for_unparker = False # need special handling for waiting for unparker

        # ============= Information of other vehicles ===========
        self.other_vehicles: Set(int) = set() # Other vehicle ids
        self.other_state: Dict[int, VehicleState] = {}
        self.other_ref_pose: Dict[int, VehiclePrediction] = {}
        self.other_ref_v: Dict[int, float] = {}
        self.other_target_idx: Dict[int, int] = {}
        self.other_crash_set: Dict[int, Set[int]] = {}
        self.other_priority: Dict[int, float] = {}
        self.other_parking_flag: Dict[int, str] = {}
        self.other_braking_flag: Dict[int, str] = {}
        self.other_waiting_for: Dict[int, int] = {}

        # ============== Method to exchange information
        self.method_to_get_central_occupancy = None
        self.method_to_change_central_occupancy = None
        self.method_to_change_other_priority = None
        self.method_to_change_other_crash_set = None

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

    def load_parking_spaces(self, parking_spaces_path: str):
        home_path = str(Path.home())
        self.parking_spaces = np.load(home_path + parking_spaces_path)

    def load_graph(self, waypoints_graph_path: str, entrance_coords: List[float]):
        """
        waypoints_graph_path: path to WaypointGraph object pickle
        entrance_coords: The (x,y) coordinates of the entrance
        """
        home_path = str(Path.home())
        with open(home_path + waypoints_graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        self.entrance_vertex = self.graph.search(entrance_coords)

    def load_maneuver(self, offline_maneuver_path: str, overshoot_ranges: Dict[str, List[Tuple[int]]]):
        home_path = str(Path.home())
        self.offline_maneuver = OfflineManeuver(pickle_file=home_path+offline_maneuver_path)

        self.overshoot_ranges = overshoot_ranges

    def load_intent_model(self, model_path: str):
        pass

    def set_anchor(self, going_to_anchor: bool=None, spot_index: int=None, should_overshoot: bool=None, anchor_points: List[int]=None, anchor_spots: List[List[int]]=None):

        if going_to_anchor is not None:
            self.going_to_anchor = going_to_anchor
        
        if spot_index is not None:
            self.spot_index = spot_index
            if self.spot_index < 0: # are we waiting to unpark or are currently unparking?
                self.parking_flag = "UNPARKING"

        if should_overshoot is not None:
            self.should_overshoot = should_overshoot
        
        if anchor_points is not None:
            self.anchor_points = anchor_points
        if anchor_spots is not None:
            self.anchor_spots = anchor_spots

    # starts from an anchor point, goes to an arbitrary spot
    def plan_from_anchor(self, new_spot_index):
        
        # go from current location to new spot
        graph_sol = AStarPlanner(self.graph.vertices[self.graph.search([self.state.x.x, self.state.x.y])], self.graph.vertices[self.graph.search(self.parking_spaces[new_spot_index])]).solve()
        new_ax = []
        new_ay = []
        for edge in graph_sol.edges:
            new_ax.append(edge.v1.coords[0])
            new_ay.append(edge.v1.coords[1])
        new_ax.append(graph_sol.edges[-1].v2.coords[0])
        new_ay.append(graph_sol.edges[-1].v2.coords[1])
        
        # do parking stuff
        
        should_overshoot = False
        
        # add the last waypoint to prepare for parking
        
        spot_x = self.parking_spaces[new_spot_index][0]
        
        if len(new_ax) == 0: # deciding to park in spot at an anchor point
            pointed_right = self.state.e.psi < np.pi / 2 and self.state.e.psi > -np.pi / 2
        else:
            last_edge = graph_sol.edges[-1]
            pointed_right = last_edge.v2.coords[0] - last_edge.v1.coords[0] > 0

        if pointed_right:
            overshoot_ranges = self.overshoot_ranges['pointed_right']
        else:
            overshoot_ranges = self.overshoot_ranges['pointed_left']

        should_overshoot = any([new_spot_index >= r[0] and new_spot_index <= r[1] for r in overshoot_ranges]) or len(new_ax) == 0

        if should_overshoot: # should overshoot
            # add point past the final waypoint, that signifies going past the spot by 4 meters, so it parks in the right place

            # if the last edge was pointed right, offset to the right
            if pointed_right:
                new_ax.append(spot_x + 4)
            else:
                new_ax.append(spot_x - 4)
        else:
            # if the last edge was pointed right, offset to the left
            last_x = spot_x - 4 if pointed_right else spot_x + 4

            last_waypoint = None
            for i in reversed(range(len(new_ax))):
                if pointed_right:
                    if new_ax[i] < last_x:
                        last_waypoint = i
                        break
                else:
                    if new_ax[i] > last_x:
                        last_waypoint = i
                        break
            new_ax, new_ay = new_ax[:last_waypoint + 1], new_ay[:last_waypoint + 1]

            if pointed_right:
                new_ax.append(spot_x - 4)
            else:
                new_ax.append(spot_x + 4)

        # have the y coordinate of the last waypoint be the same as the previous last
    
        if len(new_ax) == 0:
            new_ay.append(self.y_ref[-1])
        else:
            new_ay.append(last_edge.v2.coords[1])
        
        # offsets for lanes
        new_cx, new_cy, new_cyaw, _, _ = calc_spline_course(new_ax, new_ay, ds=0.1)
        new_cx = [new_cx[j] + self.vehicle_config.offset * np.sin(new_cyaw[j]) for j in range(len(new_cx))]
        new_cy = [new_cy[j] - self.vehicle_config.offset * np.cos(new_cyaw[j]) for j in range(len(new_cy))]
        
        # set new targets for vehicle
        self.set_ref_pose(new_cx, new_cy, new_cyaw)
        self.set_target_idx(0)
        self.set_anchor(going_to_anchor=False, spot_index=new_spot_index, should_overshoot=should_overshoot)

    def start_vehicle(self):

        assert self.parking_spaces is not None, "Please run load_parking_spaces first."
        assert self.spot_index is not None, "Please run set_anchor first."
        assert self.graph is not None, "Please run load_graph first."

        if self.spot_index > 0: # entering
            graph_sol = AStarPlanner(self.graph.vertices[self.entrance_vertex], self.graph.vertices[self.graph.search(self.parking_spaces[self.spot_index])]).solve()
        else: # exiting
            graph_sol = AStarPlanner(self.graph.vertices[self.graph.search(self.parking_spaces[-self.spot_index])], self.graph.vertices[self.entrance_vertex]).solve()

        x_ref, y_ref, yaw_ref = graph_sol.compute_ref_path(self.vehicle_config.offset)

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
        # return self.last_idx == self.target_idx
        # need to constantize this
        if not self.reached_tgt:
            dist = np.linalg.norm([self.state.x.x - self.x_ref[-1], self.state.x.y - self.y_ref[-1]])
            ang = ((np.arctan2(self.y_ref[-1] - self.state.x.y, self.x_ref[-1] - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
            self.reached_tgt = dist < 5 and ang > (np.pi / 2) and ang < (3 * np.pi / 2)
            # self.reached_tgt = np.linalg.norm([self.state.x.x - self.x_ref[-1], self.state.x.y - self.y_ref[-1]]) < threshold
        return self.reached_tgt

    def num_waypoints(self):
        return len(self.x_ref)

    def set_method_to_get_central_occupancy(self, method):
        self.method_to_get_central_occupancy = method

    def set_method_to_change_central_occupancy(self, method):
        self.method_to_change_central_occupancy = method

    def set_method_to_change_other_priority(self, method):
        self.method_to_change_other_priority = method

    def set_method_to_change_other_crash_set(self, method):
        self.method_to_change_other_crash_set = method

    def get_central_occupancy(self):
        """
        Get the parking spaces and occupancy
        """
        method = self.method_to_get_central_occupancy
        if callable(method):
            # Call ROS service to get occupancy
            self.occupancy = method()
        else:
            # Otherwise, for testing, method is occupancy from simulator
            self.occupancy = method

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

    def change_other_crash_set(self, other_id: int, data_to_change: int, operation: str):
        # If we are adding things back
        if other_id == self.vehicle_id:
            if operation == "add":
                self.crash_set.add(data_to_change)
            elif operation == "remove":
                self.crash_set.discard(data_to_change)
            else:
                raise Exception("Unrecognized operation type. Choose from 'add' and 'remove' ")
            return

        self.other_crash_set[other_id].add(self.vehicle_id)

        method = self.method_to_change_other_crash_set

        if callable(method):
            # With ROS, add_method will be a function to call ROS service
            method(other_id, data_to_change, operation)
        else:
            # Without ROS, we are directly changing the object's value, then method is actually a dict of [id, Vehicle object]
            if operation == "add":
                method[other_id].crash_set.add(data_to_change)
            elif operation == "remove":
                method[other_id].crash_set.discard(data_to_change)
            else:
                raise Exception("Unrecognized operation type. Choose from 'add' and 'remove' ")


    def change_other_priority(self, other_id: int, new_value: int):
        # If we are changing priority for ourselves
        if other_id == self.vehicle_id:
            self.priority = new_value
            return

        self.other_priority[other_id] = new_value

        method = self.method_to_change_other_priority
        if type(method) == function:
            # With ROS, add_method will be a function to call ROS service
            method(other_id, new_value)
        else:
            # Without ROS, we are directly changing the object's value, then adding_method is actually a dict of [id, Vehicle object]
            method[other_id].priority = new_value

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
            self.other_crash_set[id] = v.crash_set
            self.other_priority[id] = v.priority
            self.other_parking_flag[id] = v.parking_flag
            self.other_braking_flag[id] = v.braking_flag
            self.other_waiting_for[id] = v.waiting_for

    def will_crash(self) -> Set[int]:
        surrounding_ids = [id for id in self.other_vehicles if np.linalg.norm([self.state.x.x - self.other_state[id].x.x, self.state.x.y - self.other_state[id].x.y]) < self.vehicle_config.crash_check_radius]

        will_crash_with = set()

        # create states for looking ahead
        look_ahead_state = self.state.copy()
        other_look_ahead_states = [self.other_state[id].copy() for id in surrounding_ids]

        # for each time step, looking ahead
        for _ in range(self.vehicle_config.look_ahead_timesteps):
            # calculate new positions
            self.motion_predictor.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
            self.motion_predictor.set_ref_v(self.v_ref)
            self.motion_predictor.set_target_idx(self.target_idx)
            ai, di, _ = self.motion_predictor.solve(look_ahead_state, self.is_braking())
            self.motion_predictor.step(look_ahead_state, ai, di)

            for id, other_look_ahead_state in zip(surrounding_ids, other_look_ahead_states):
                if id not in will_crash_with: # for efficiency
                    self.motion_predictor.set_ref_pose(self.other_ref_pose[id].x, self.other_ref_pose[id].y, self.other_ref_pose[id].psi)
                    self.motion_predictor.set_ref_v(self.other_ref_v[id])
                    self.motion_predictor.set_target_idx(self.other_target_idx[id])
                    ai, di, _ = self.motion_predictor.solve(other_look_ahead_state, self.other_braking_flag != "NOT_BRAKING")
                    self.motion_predictor.step(other_look_ahead_state, ai, di)


            # detect crash
            for id, other_look_ahead_state in zip(surrounding_ids, other_look_ahead_states):
                if id not in will_crash_with: # for efficiency
                    if self.will_collide(other_look_ahead_state, self.vehicle_body):
                        # TODO: Here we assume all other vehicles have the same vehicle body as us
                        will_crash_with.add(id)

        return will_crash_with

    def update_state(self):
        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        self.controller.set_ref_v(self.v_ref)
        self.controller.set_target_idx(self.target_idx)
        # get acceleration toward target speed (ai), amount we should turn (di), and next target (target_idx)
        ai, di, self.target_idx = self.controller.solve(self.state, self.is_braking())
        # advance state of vehicle (updates x, y, yaw, velocity)
        self.controller.step(self.state, ai, di)
            
    def update_state_parking(self, advance=True):
        if self.parking_maneuver_state is None: # start parking
            # get parking parameters
            direction = 'west' if self.state.e.psi > np.pi / 2 or self.state.e.psi < -np.pi / 2 else 'east'
            if self.should_overshoot:
                location = 'right' if (direction == 'east') else 'left' # we are designed to overshoot the spot
            else:
                location = 'left' if (direction == 'east') else 'right' # we are designed to undershoot the spot
            pointing = 'up' if np.random.rand() < 0.5 else 'down' # random for diversity
            north_spot_ranges = [(0, 41), (67, 91), (113, 133), (159, 183), (205, 225), (251, 275), (297, 317)]
            spot = 'north' if any([self.spot_index >= r[0] and self.spot_index <= r[1] for r in north_spot_ranges]) else 'south'
            
            # get parking maneuver
            self.parking_maneuver_state, self.parking_maneuver_input = self.offline_maneuver.get_maneuver([self.park_start_coords[0] - 4 if location == 'right' else self.park_start_coords[0] + 4, self.park_start_coords[1]], direction, location, spot, pointing)
            
            
        # idle when done
        step = min(self.parking_step, len(self.parking_maneuver_state['x']) - 1)
            
        # set state
        self.state.x.x = self.parking_maneuver_state['x'][step]
        self.state.x.y = self.parking_maneuver_state['y'][step]
        self.state.e.psi = self.parking_maneuver_state['yaw'][step]
        self.state.v.v = self.parking_maneuver_state['v'][step]
        
        # update parking step if advancing
        self.parking_step += 1 if advance else 0
        
    def update_state_unparking(self, advance=True):
        if self.unparking_maneuver_state is None: # start unparking
            # get unparking parameters
            direction = 'west' if self.x_ref[0] > self.x_ref[1] else 'east' # if first direction of travel is left, face west
            location = 'right' if np.random.rand() < 0.5 else 'left' # random for diversity
            pointing = 'up' if self.state.e.psi > 0 else 'down' # determine from state
            north_spot_ranges = [(0, 41), (67, 91), (113, 133), (159, 183), (205, 225), (251, 275), (297, 317)]
            spot = 'north' if any([-self.spot_index >= r[0] and -self.spot_index <= r[1] for r in north_spot_ranges]) else 'south'
            
            # get parking maneuver
            self.unparking_maneuver_state, self.unparking_maneuver_input = self.offline_maneuver.get_maneuver([self.state.x.x if location == 'right' else self.state.x.x, self.state.x.y - 6.25 if spot == 'north' else self.state.x.y + 6.25], direction, location, spot, pointing)
            
            # set initial unparking state
            self.unparking_step = len(self.unparking_maneuver_state['x']) - 1
            
        # get step
        step = self.unparking_step
            
        # set state
        self.state.x.x = self.unparking_maneuver_state['x'][step]
        self.state.x.y = self.unparking_maneuver_state['y'][step]
        self.state.e.psi = self.unparking_maneuver_state['yaw'][step]
        self.state.v.v = self.unparking_maneuver_state['v'][step]
        
        if self.unparking_step == 0: # done unparking
            self.parking_flag = None
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

        center = np.array([state.x.x, state.x.y])
        psi = state.e.psi

        R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

        corners = (R @ vehicle_body.V.T).T

        return corners + center

    def brake(self, braking_flag="BRAKING"):
        """
        Set target speed to 0 and turn on brakes, which make deceleration faster
        """
        self._pre_brake_target_speed = self.v_ref
        self.v_ref = 0
        self.braking_flag = braking_flag

    def unbrake(self):
        """
        Set target speed back to what it was. Only does something if braking
        """
        if self.braking_flag != "NOT_BRAKING":
            self.v_ref = self._pre_brake_target_speed
            self.braking_flag = "NOT_BRAKING"
            self.crash_set.clear()
            self.priority = None
            self.waiting_for = None

    def is_braking(self):
        """
        Are we braking?
        """
        return self.braking_flag != "NOT_BRAKING"
    
    def is_unparking(self):
        """
        Have we started the unparking maneuver yet? If this is False, traffic should have the right of way, else this vehicle should have the right of way
        """
        # TODO: Can we delete this?
        return self.parking_flag == "UNPARKING" and self.unparking_maneuver_state is not None and self.unparking_step < len(self.unparking_maneuver_state['x']) - 1
    
    def is_all_done(self):
        """
        Have we finished the parking maneuver or we have reached the exit?
        """
        
        return (self.parking_flag == "PARKING" and self.parking_maneuver_state is not None and self.parking_step >= len(self.parking_maneuver_state['x'])) or (self.spot_index < 0 and self.reached_tgt)

    def solve(self):
        """
        Having other_vehicle_objects here is just to mimic the ROS service to change values of the other vehicle. Should use this to acquire information
        """
        # driving control

        if self.parking_flag:
            pass
        elif self.going_to_anchor or not self.reached_target():
            # normal driving (haven't reached pre-parking point)

            # braking controller
            if self.braking_flag == "NOT_BRAKING":
                # normal speed controller if not braking
                if self.target_idx < self.num_waypoints() - self.vehicle_config.steps_to_end:
                    self.set_ref_v(self.vehicle_config.v_max)
                else:
                    self.set_ref_v(self.vehicle_config.v_end)

                # detect parking and unparking
                nearby_parkers = []
                for id in self.other_vehicles:
                    condition = self.other_parking_flag[id] \
                                and np.linalg.norm([self.state.x.x - self.other_state[id].x.x, self.state.x.y - self.other_state[id].x.y]) < self.vehicle_config.parking_radius

                    if condition:
                        nearby_parkers.append(id)

                if nearby_parkers:
                    # should only be one nearby parker, since they wait for each other
                    # TODO: What does this mean?
                    parker_id = nearby_parkers[0]
                    self.brake(braking_flag="WAITING")
                    self.waiting_for = parker_id
                    self.priority = -1

                    if self.other_parking_flag[parker_id] == "UNPARKING":
                        self.waiting_for_unparker = True

                else: # No one is parking
                    self.crash_set.update(self.will_crash())

                    # If will crash
                    if self.crash_set:
                        # add ourselves to the crash set
                        self.crash_set.add(self.vehicle_id)

                        # add ourselves to other vehicle crash sets (to cause them to stop)
                        for id in self.crash_set:
                            self.change_other_crash_set(id, self.vehicle_id, "add")

                        # recursively add all that they will also crash with to our set
                        secondary_crash_set = set()
                        old_len = 0
                        new_len = 1 # just to make sure the loop runs at least once

                        # keep checking until no longer adding vehicles
                        while new_len - old_len > 0:
                            old_len = len(secondary_crash_set)
                            for id in self.crash_set:
                                if id == self.vehicle_id:
                                    continue

                                secondary_crash_set.update(self.other_crash_set[id])
                            new_len = len(secondary_crash_set)
                            self.crash_set.update(secondary_crash_set)

                        # variable to tell where to go next (default is braking)
                        next_flag = "BRAKING"

                        # set priority if not already set for this vehicle
                        if self.priority is None:
                            leading_trailing_scenario = False
                            # for two vehicle leading/trailing situation
                            if len(self.crash_set) == 2:
                                other_id = None
                                for id in self.crash_set:
                                    if id != self.vehicle_id:
                                        other_id = id

                                ang = (self.state.e.psi - self.other_state[other_id].e.psi + (2*np.pi)) % (2*np.pi) # [0, 2pi)

                                thres = self.vehicle_config.leading_trailing_thres

                                if ang < thres or ang > 2*np.pi - thres:
                                    leading_trailing_scenario = True

                                    this_to_other_ang = ((np.arctan2(self.other_state[other_id].x.y - self.state.x.y, self.other_state[other_id].x.x - self.state.x.x) - self.other_state[other_id].e.psi) + (2*np.pi)) % (2*np.pi)

                                    if this_to_other_ang < np.pi / 2 or this_to_other_ang > 3 * np.pi / 2: # trailing car
                                        next_flag = "WAITING" # go straight to waiting, no priority calculations necessary
                                        self.priority = self.other_priority[other_id] - 1 if self.other_priority[other_id] is not None else -1 # so cars that may brake behind it can have a priority
                                        self.waiting_for = other_id
                                    else: # leading car
                                        next_flag = "NOT_BRAKING" # don't brake
                                        self.crash_set.clear() # not going to crash anymore 

                            if not leading_trailing_scenario: # There are more than two vehicles 
                                
                                if all([self.other_braking_flag[id]]=="NOT_BRAKING" for id in self.crash_set if id != self.vehicle_id):
                                    # if this is first detection of collision
                                    # NOTE: any priorities set in here should be between 0 (inclusive) and 1 (exclusive)
                                    for id in self.crash_set:
                                        self.change_other_priority(id, np.random.rand())
                                else: # new car meeting up with cars that have already braked
                                    # wait for last car in queue
                                    next_flag = "WAITING"
                                    self.waiting_for = min([id for id in self.crash_set if id != self.vehicle_id], key=lambda id: self.other_priority[id])

                                    self.priority = min([self.other_priority[id] for id in self.crash_set if id != self.vehicle_id]) - 1
                        
                        if next_flag != "NOT_BRAKING":
                            self.brake(braking_flag=next_flag)

            elif self.braking_flag == "BRAKING": # when we don't know who we're waiting for yet, but know we need to brake
                
                # don't check for crash with self, or vehicles that are all done
                crasher_ids = self.will_crash()

                if crasher_ids:
                    self.crash_set.update(crasher_ids)

                    for other_id in crasher_ids:
                        self.change_other_crash_set(other_id, self.vehicle_id, "add")

                # recursively add all that they will also crash with to our set
                secondary_crash_set = set()
                old_len = 0
                new_len = 1 # just to make sure the loop runs at least once

                # keep checking until no longer adding vehicles
                while new_len - old_len > 0:
                    old_len = len(secondary_crash_set)
                    for id in self.crash_set:
                        secondary_crash_set.update(self.other_crash_set[id])
                    new_len = len(secondary_crash_set)
                    self.crash_set.update(secondary_crash_set)

                # if everyone stopped
                if all([self.other_state[other_id].v.v < 0.05 for other_id in self.crash_set if other_id != self.vehicle_id] + [self.state.v.v < 0.05]):
                    d = {}
                    d[self.vehicle_id] = self.priority
                    for id in self.crash_set:
                        if id != self.vehicle_id:
                            d[id] = self.other_priority[id]

                    # determine order of going (for now random)
                    order = sorted(d.items(), key=lambda o: o[1], reverse=True)

                    # determine who we're waiting for
                    if order[0][0] == self.vehicle_id: # we go first
                        self.waiting_for = None
                    else: # we're waiting
                        for oi in range(1, len(order)):
                            if order[oi][0] == self.vehicle_id:
                                self.waiting_for = order[oi - 1][0]
                    
                    self.braking_flag = "WAITING"

            else: # waiting
                # parking
                if self.waiting_for is not None and self.other_parking_flag[self.waiting_for]:
                    if self.waiting_for not in self.other_vehicles:
                        self.unbrake()
                        
                elif self.waiting_for is not None and self.waiting_for_unparker:
                    if self.other_parking_flag[self.waiting_for] != "UNPARKING":
                        self.waiting_for_unparker = False
                        self.unbrake()

                else:
                    # other (standard) cases

                    should_unbrake = False
                    # go if going first
                    if self.waiting_for is None:
                        should_unbrake = True
                    else:
                        # TODO: better heuristic for unbraking
                        ang = ((np.arctan2(self.other_state[self.waiting_for].x.y - self.state.x.y, self.other_state[self.waiting_for].x.x - self.state.x.x) - self.other_state[self.waiting_for].e.psi) + (2*np.pi)) % (2*np.pi)
                        
                        if (self.waiting_for not in self.other_vehicles  
                            or self.other_braking_flag[self.waiting_for] == "NOT_BRAKING" 
                            and ( (np.pi/2) < ang < (3*np.pi)/2) 
                            or np.linalg.norm([self.other_state[self.waiting_for].x.x - self.state.x.x, self.other_state[self.waiting_for].x.y - self.state.x.y]) > 10):
                            # TODO: Why this is 10?
                            should_unbrake = True
                        elif self.other_waiting_for[self.waiting_for] == self.vehicle_id: # if the vehicle you're waiting for is waiting for you
                            # you should go
                            # TODO: this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                            # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                            should_unbrake = True

                    if should_unbrake:
                        # remove vehicle from other vehicle's crash sets
                        for other_id in self.crash_set:
                            if other_id != self.vehicle_id and self.vehicle_id in self.other_crash_set[other_id]:
                                self.change_other_crash_set(other_id, self.vehicle_id, "remove")

                        self.unbrake() # also sets brake_state to NOT_BRAKING

            # if gotten near anchor point, figure out where to go next
            if self.going_to_anchor and self.spot_index > 0 and self.target_idx > self.num_waypoints() - 10:
                # TODO: This "10" should be replaced
                self.get_central_occupancy()

                possible_spots = [i for i in self.anchor_spots[self.anchor_points.index(self.spot_index)] if not self.occupancy[i]]

                new_spot_index = np.random.choice(possible_spots)
                self.plan_from_anchor(new_spot_index)
                
                self.occupancy[new_spot_index] = True
                self.change_central_occupancy(new_spot_index, True)
        else:
            # if reached target (pre-parking point), start parking
            self.set_ref_v(0)
            if self.spot_index > 0:
                self.parking_flag = "PARKING"

        if self.parking_flag == "PARKING":
            # wait for coast to be clear, then start parking
            # everyone within range should be braking or parking or unparking
            # TODO: this doesn't yet account for a braked vehicle in the way of our parking
            should_go = all([self.other_braking_flag[id] 
                            or self.other_parking_flag[id] 
                            or np.linalg.norm([self.state.x.x - self.other_state[id].x.x, self.state.x.y - self.other_state[id].x.y]) >= self.vehicle_config.parking_radius for id in self.other_vehicles])

            if self.park_start_coords is None:
                self.park_start_coords = (self.state.x.x - self.vehicle_config.offset * np.sin(self.state.e.psi), self.state.x.y + self.vehicle_config.offset * np.cos(self.state.e.psi))
            self.update_state_parking(should_go)
        elif self.parking_flag == "UNPARKING": # wait for coast to be clear, then start unparking
            # everyone within range should be braking or parking or unparking

            # TODO: this doesn't yet account for a braked / (un)parking vehicle in the way of our parking
            should_go = all([self.other_braking_flag[id] 
                            or self.other_parking_flag[id] 
                            or np.linalg.norm([self.state.x.x - self.other_state[id].x.x, self.state.x.y - self.other_state[id].x.y]) >= self.vehicle_config.parking_radius for id in self.other_vehicles])

            self.update_state_unparking(should_go)
        else: 
            self.update_state()

    def predict_intent(self, vehicle_id=None, radius=None):
        """
        docstring
        """
        if vehicle_id is None:
            pass
            # Predicting ourselves
        else:
            pass
            # Predicting someone else

        # The list of other vehicle ids can be found in self.other_vehicles
        # Using self.other_state[vehicle_id] to get the state of other vehicles. State is of type VehicleState

        # Filter the states that is inside the radius
        
        # Construct the instance centric view
        # Run through the intent_predictor
        # Return prob at each spot location