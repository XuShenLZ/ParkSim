from typing import Dict, List, Set, Tuple
from matplotlib.pyplot import hist
import numpy as np
from pathlib import Path
import pickle
import time
import array
from collections import deque
import torch

from parksim.path_planner.offline_maneuver import OfflineManeuver
from parksim.path_planner.offset_offline_maneuver import OffsetOfflineManeuver

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController

from parksim.pytypes import VehiclePrediction, VehicleState
from parksim.route_planner.a_star import AStarGraph, AStarPlanner
from parksim.route_planner.graph import Vertex, WaypointsGraph
from parksim.utils.get_corners import get_vehicle_corners, get_vehicle_corners_from_dict, rectangle_to_polytope
from parksim.utils.interpolation import interpolate_states_inputs
from parksim.vehicle_types import VehicleBody, VehicleConfig, VehicleInfo, VehicleTask
from parksim.intent_predict.cnn.predictor import Predictor, PredictionResponse
from parksim.intent_predict.cnn.visualizer.instance_centric_generator import InstanceCentricGenerator

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

class RuleBasedStanleyVehicle(AbstractAgent):
    def __init__(self, vehicle_id: int, vehicle_body: VehicleBody, vehicle_config: VehicleConfig, controller: StanleyController = StanleyController(), motion_predictor: StanleyController = StanleyController(), inst_centric_generator = InstanceCentricGenerator(), intent_predictor = Predictor(), traj_model = None, intent_extractor: CNNDataProcessor = None, traj_extractor: TransformerDataProcessor = None, spot_detector: LocalDetector = None, electric_vehicle: bool = False, intent_vehicle: bool = False):
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

        self.task_profile: List[VehicleTask] = []
        self.task_history: List[VehicleTask] = []
        self.current_task: str = None

        self.idle_duration = None
        self.idle_end_time = None
        self.idle_start_time = None

        # Dimensions
        self.vehicle_body = vehicle_body

        self.vehicle_config = vehicle_config

        # Controller and predictor
        self.controller = controller
        self.motion_predictor = motion_predictor
        self.intent_predictor = intent_predictor # cnnV2 
        self.inst_centric_generator = inst_centric_generator

        self.target_idx = 0
        
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

        self.parking_start_time = float('inf') # inf means haven't start parking or unparking. Anything above 0 is parking

        self.parking_maneuver = None
        self.parking_step = 0
        
        # unparking stuff
        self.unparking_maneuver = None
        self.unparking_step = -1
        self.unparking_x_ref = None
        self.unparking_y_ref = None
        self.unparking_yaw_ref = None
        
        # braking stuff
        self.is_braking = False # are we braking?
        self.last_braking_distance = None # to prevent braking because will crash, then immediate unbraking because not inside braking distance
        self._pre_brake_target_speed = 0 # speed to restore when unbraking
        self.priority = 0 # priority for going after braking
        self.waiting_for: int = 0 # vehicle waiting for before we go. We start indexing vehicles from 1, so 0 means no vehicle
        self.waiting_for_unparker = False # need special handling for waiting for unparker

        # ev charging
        self.charging_spots = [39, 40, 41] # TODO: put this in a yaml
        self.ev = electric_vehicle
        self.ev_charging_state = None # none = not ev, 0 = waiting to charge, 1 = charging, 2 = done charging TODO: make enum

        # intent predict
        self.intent_vehicle = intent_vehicle

        self.loops = 0
        self.loops_before_predict = 5
        self.loops_between_predict = 5

        self.intent = None

        self.traj_model = traj_model
        self.mode='v1'

        self.intent_extractor = intent_extractor 
        self.traj_extractor = traj_extractor

        self.spot_detector = spot_detector
        self.solver = pyo.SolverFactory('ipopt')

        self.prediction_history = {}
        self.input_history = {}
        self.offset_offline_maneuver = None
        self.offset_parking_maneuver = None
        self.intent_spot = None
        self.intent_parking_step = None
        self.intent_parking_origin = None # (x, y, psi)
        self.intent_parking_start_time = None

        self.logger = deque(maxlen=100)

        # ============= Information of other vehicles ===========
        self.other_vehicles: Set(int) = set() # Other vehicle ids
        self.nearby_vehicles: Set(int) = set() # Nearby vehicles that we are interested
        self.other_state: Dict[int, VehicleState] = {}
        self.other_ref_pose: Dict[int, VehiclePrediction] = {}
        self.other_ref_v: Dict[int, float] = {}
        self.other_target_idx: Dict[int, int] = {}
        self.other_priority: Dict[int, int] = {}
        self.other_task: Dict[int, str] = {} # The current task of other vehicle
        self.other_parking_progress: Dict[int, str] = {} # Other vehicles will broadcast "PARKING" if vehicle.is_parking(), "UNPARKING" if vehicle.is_unparking(), None otherwise
        self.other_parking_start_time: Dict[int, float] = {}
        self.other_is_braking: Dict[int, str] = {}
        self.other_waiting_for: Dict[int, int] = {}
        self.other_is_all_done: Dict[int, bool] = {}

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

    def set_vehicle_state(self, state: VehicleState = None, spot_index: int = None, heading: float = None):
        if state is not None:
            self.state = state
        elif spot_index is not None:
            assert self.parking_spaces is not None, "Please run load_parking_spaces first."

            self.spot_index = spot_index

            self.state.x.x = self.parking_spaces[spot_index][0]
            self.state.x.y = self.parking_spaces[spot_index][1]
            if heading is not None:
                self.state.e.psi = heading
            else:
                self.state.e.psi = np.pi / 2 if np.random.rand() < 0.5 else -np.pi / 2

    def set_task_profile(self, task_profile):
        self.task_profile = task_profile

        # ev charging
        if self.ev:
            for task in task_profile:
                if task.name == 'PARK' and task.target_spot_index in self.charging_spots: # will park in charging spot
                    self.ev_charging_state = 0
            if self.ev_charging_state is None: # already done
                self.ev_charging_state = 2

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

        # Default entrance vertex
        self.entrance_vertex = self.graph.search(entrance_coords)

    def load_maneuver(self, offline_maneuver_path: str):
        home_path = str(Path.home())
        self.offline_maneuver = OfflineManeuver(pickle_file=home_path+offline_maneuver_path)

    def load_offset_maneuver(self, offline_maneuver_path: str):
        home_path = str(Path.home())
        self.offset_offline_maneuver = OffsetOfflineManeuver(pickle_file=home_path+offline_maneuver_path)

    def load_intent_model(self, model_path: str):
        """
        load_graph must be called before load_intent_model.
        """
        home_path = str(Path.home())
        self.intent_predictor.load_model(waypoints=self.graph, model_path=home_path + model_path)
    
    def compute_ref_path(self, graph_sol: AStarGraph, offset: float = None, spot_index: int = None):
        if not offset:
            offset = self.vehicle_config.offset

        if spot_index is None:
            # exiting
            if len(graph_sol.vertices) == 0: # just point right by default

                start_coords = np.array([self.state.x.x, self.state.x.y])
                start_vertex_idx = self.graph.search(start_coords)
                graph_sol.vertices = [Vertex(start_coords), self.graph.vertices[start_vertex_idx]]

            x_ref, y_ref, yaw_ref = graph_sol.compute_ref_path(offset)
        else:

            if len(graph_sol.vertices) == 0: # just point right by default

                start_coords = np.array([self.state.x.x, self.state.x.y])
                start_vertex_idx = self.graph.search(start_coords)
                last_x, last_y = self.graph.vertices[start_vertex_idx].coords
                graph_sol.vertices = [Vertex(np.array([last_x, last_y])), Vertex(np.array([last_x + 4, last_y]))]

            elif len(graph_sol.vertices) == 1: # just point right by default
                
                last_x, last_y = graph_sol.vertices[0].coords
                graph_sol.vertices.append([Vertex(np.array([last_x + 4, last_y]))])

            else:

                # parking
                last_edge = graph_sol.edges[-1]
                pointed_right = last_edge.v2.coords[0] - last_edge.v1.coords[0] > 0

                if pointed_right:
                    overshoot_ranges = self.overshoot_ranges['pointed_right']
                else:
                    overshoot_ranges = self.overshoot_ranges['pointed_left']

                self.should_overshoot = any([spot_index >= r[0] and spot_index <= r[1] for r in overshoot_ranges])

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

                    last_waypoint = len(graph_sol.vertices) - 1
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

    def cruise_planning(self, task: VehicleTask):

        assert self.parking_spaces is not None, "Please run load_parking_spaces first."
        # assert self.spot_index is not None, "Please run set_spot_idx first."
        assert self.graph is not None, "Please run load_graph first."

        self.vehicle_config.v_cruise = task.v_cruise

        start_coords = np.array([self.state.x.x, self.state.x.y])

        start_vertex_idx = self.graph.search(start_coords)

        if task.target_spot_index is not None:
            # Going to a spot
            
            is_north_spot = any([abs(task.target_spot_index) >= r[0] and abs(
                task.target_spot_index) <= r[1] for r in self.north_spot_idx_ranges])
            y_offset = -self.spot_y_offset if is_north_spot else self.spot_y_offset
            waypoint_coords = [self.parking_spaces[abs(
                task.target_spot_index)][0], self.parking_spaces[abs(task.target_spot_index)][1] + y_offset]

            graph_sol = AStarPlanner(
                self.graph.vertices[start_vertex_idx], self.graph.vertices[self.graph.search(waypoint_coords)]).solve()

            x_ref, y_ref, yaw_ref = self.compute_ref_path(
                graph_sol=graph_sol, spot_index=task.target_spot_index)

        elif task.target_coords is not None:
            # Travel to a coordinates
            graph_sol = AStarPlanner(
                self.graph.vertices[start_vertex_idx], self.graph.vertices[self.graph.search(task.target_coords)]).solve()

            if len(graph_sol.edges) == 0: # just go to a waypoint
                x_ref = [self.state.x.x, task.target_coords[0]]
                y_ref = [self.state.x.y, task.target_coords[1]]
                yaw_ref = [self.state.e.psi, self.state.e.psi]
            else: # compute a-star path
                x_ref, y_ref, yaw_ref = self.compute_ref_path(graph_sol=graph_sol, spot_index=None)

        self.set_ref_pose(x_ref, y_ref, yaw_ref)
        self.set_ref_v(0)

    def execute_next_task(self):
        if len(self.task_profile) > 0:
            task = self.task_profile.pop(0)

            self.current_task = task.name

            if task.name == "CRUISE":
                if task.target_spot_index is not None:
                    self.spot_index = task.target_spot_index
                    
                if self.unparking_x_ref is None:
                    self.cruise_planning(task=task)
                else:
                    self.unparking_x_ref = None
                    self.unparking_y_ref = None
                    self.unparking_yaw_ref = None 
            elif task.name == "PARK":
                self.spot_index = task.target_spot_index
            elif task.name == "UNPARK":
                pass

                if self.task_profile[0].name == "CRUISE":
                    # Need to try the next CRUISE task for getting the direction to unpark
                    self.cruise_planning(self.task_profile[0])
                    self.unparking_x_ref, self.unparking_y_ref, self.unparking_yaw_ref = self.x_ref, self.y_ref, self.yaw_ref
                else:
                    raise ValueError("UNPARK task should be followed with a CRUISE task.")

                if self.ev and task.target_spot_index in self.charging_spots:
                    self.ev_charging_state = 2

            elif task.name == "IDLE":
                if task.duration is not None:
                    self.idle_duration = task.duration
                else:
                    self.idle_end_time = task.end_time

                # ev charging
                # if charging, set variable
                if self.ev and len(self.task_profile) > 0 and self.task_profile[0].name == 'UNPARK' and self.task_profile[0].target_spot_index in self.charging_spots:
                    self.ev_charging_state = 1
            else:
                raise ValueError(f'Undefined task name. {task.name} is received.')

            self.task_history.append(task)
        else:
            # Finished all tasks
            self.current_task = "END"
    
    def reached_target(self, target=None):
        if target is None:
            target = [self.x_ref[-1], self.y_ref[-1]]
        dist = np.linalg.norm([self.state.x.x - target[0], self.state.x.y - target[1]])
        ang = ((np.arctan2(target[1] - self.state.x.y, target[0] - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
        reached_tgt = dist < self.vehicle_config.braking_distance/2 and ang > (np.pi / 2) and ang < (3 * np.pi / 2)
        return reached_tgt

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
        self.info.task = self.current_task

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
        self.info.is_all_done = self.is_all_done()

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

            self.other_task[id] = v.current_task
            if v.is_parking():
                self.other_parking_progress[id] = "PARKING"
            elif v.is_unparking():
                self.other_parking_progress[id] = "UNPARKING"
            else:
                self.other_parking_progress[id] = ""

            self.other_is_braking[id] = v.is_braking
            self.other_parking_start_time[id] = v.parking_start_time
            self.other_waiting_for[id] = v.waiting_for
            self.other_is_all_done[id] = v.is_all_done()

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
        parking_dist_away: additional check, if this_id's x-coordinate is parking_dist_away past other_id's x-coordinate 
        """
        if this_id is None or this_id == self.vehicle_id:
            this_corners = self.get_corners()
            this_state = self.state
            this_psi = (this_state.e.psi + (2*np.pi)) % (2*np.pi)
        else:
            this_corners = self.get_corners(self.other_state[this_id])
            this_state = self.other_state[this_id]
            this_psi = (this_state.e.psi + (2*np.pi)) % (2*np.pi)
        
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
            if this_psi > np.pi / 2 and this_psi < np.pi * 3 / 2: # facing east 
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

            self.change_central_occupancy(self.spot_index, True)
            
            
        step = self.parking_step
        # set state
        self.state.x.x = self.parking_maneuver.x[step]
        self.state.x.y = self.parking_maneuver.y[step]
        self.state.e.psi = self.parking_maneuver.psi[step]
        self.state.v.v = self.parking_maneuver.v[step]

        self.state.u.u_a = self.parking_maneuver.u_a[step]
        self.state.u.u_steer = self.parking_maneuver.u_steer[step]
        
        if self.parking_step >= len(self.parking_maneuver.x) - 1:
            # done parking
            self.reset_parking_related()

            self.execute_next_task()
        else:
            # update parking step if advancing
            self.parking_step += 1 if advance else 0
        
    def update_state_unparking(self, advance=True):
        if self.unparking_maneuver is None: # start unparking
            # get unparking parameters
            direction = 'west' if self.x_ref[0] >= self.x_ref[1] and self.spot_index not in [91, 183, 275] else 'east' # if first direction of travel is left, face west
            location = 'right' if np.random.rand() < 0.5 else 'left' # random for diversity
            pointing = 'up' if self.state.e.psi > 0 else 'down' # determine from state
            spot = 'north' if any([abs(self.spot_index) >= r[0] and abs(self.spot_index) <= r[1] for r in self.north_spot_idx_ranges]) else 'south'
            
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
            self.change_central_occupancy(self.spot_index, False)
            
            self.reset_parking_related()
            self.execute_next_task()
        else:
            # update parking step if advancing
            self.unparking_step -= 1 if advance else 0

    def reset_parking_related(self):
        # inf means haven't start parking or unparking. Anything above 0 is parking
        self.parking_start_time = float('inf')

        self.parking_maneuver = None
        self.parking_step = 0

        # unparking stuff
        self.unparking_maneuver = None
        self.unparking_step = -1

        self.park_start_coords = None

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
        if self.waiting_for != 0:
            self.last_braking_distance = self.dist_from(self.waiting_for)

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
        return self.current_task == "PARK" and self.parking_maneuver is not None and self.parking_step > 0 and self.parking_step < len(self.parking_maneuver.x) - 1

    def is_unparking(self):
        return (self.current_task == "UNPARK" and self.unparking_maneuver is not None and self.unparking_step < len(self.unparking_maneuver.x) - 1 and self.unparking_step > 0) and not (self.current_task == "UNPARK" and self.unparking_step == -1)
    
    def is_all_done(self):
        """
        Have we finished the parking maneuver or we have reached the exit?
        """
        
        return self.current_task == "END"
    
    def update_nearby_vehicles(self, radius=None):
        """
        radius: (Optional) vehicles inside this radius are considered as "nearby". If left empty, we will use vehicle_config.crash_check_radius
        """
        if not radius:
            radius = self.vehicle_config.crash_check_radius

        self.nearby_vehicles.clear()

        for id in self.other_vehicles:
            if self.other_is_all_done[id]:
                continue
            
            dist = self.dist_from(id)

            if dist < radius:
                self.nearby_vehicles.add(id)

    def solve(self, time=None, timer_period=None, other_vehicles=[], history=None, coord_spot_fn=None, obstacle_corners={}):
        if self.intent_vehicle:
            self.solve_intent_driving(time=time, timer_period=timer_period, other_vehicles=other_vehicles, history=history, coord_spot_fn=coord_spot_fn, obstacle_corners=obstacle_corners)
        else:
            self.solve_classic(time)    

    def solve_classic(self, time=None):
        """
        Having other_vehicle_objects here is just to mimic the ROS service to change values of the other vehicle. Should use this to acquire information
        """
        if self.current_task == "END":
            return

        # Firstly update the set of nearby_vehicles
        self.update_nearby_vehicles()

        # driving control

        if self.current_task in ["PARK", "UNPARK"]:
            pass
        elif self.current_task == "IDLE":
            if self.idle_start_time is None:
                self.idle_start_time = time
            
            if self.idle_duration is not None and time - self.idle_start_time >= self.idle_duration \
                or self.idle_end_time is not None and time >= self.idle_end_time:
                self.idle_start_time = None
                self.idle_duration = None
                self.idle_end_time = None
                self.execute_next_task()
        elif not self.reached_target():
            # normal driving (haven't reached pre-parking point)

            # braking controller
            if not self.is_braking:
                # normal speed controller if not braking
                if self.target_idx < self.num_waypoints() - self.vehicle_config.steps_to_end:
                    self.set_ref_v(self.vehicle_config.v_cruise)
                else:
                    self.set_ref_v(self.vehicle_config.v_end)

                # detect parking and unparking
                nearby_parkers = [id for id in self.nearby_vehicles if self.other_parking_progress[id] and self.other_within_parking_box(id) and not self.has_passed(other_id=id, parking_dist_away=2)]

                if nearby_parkers:
                    # should only be one nearby parker, since they wait for each other
                    parker_id = nearby_parkers[0]
                    self.waiting_for = parker_id
                    self.brake()
                    self.priority = -1

                    if self.other_task[parker_id] == "UNPARK":
                        self.waiting_for_unparker = True

                else: # No one is parking
                    ids_will_crash_with = list(self.will_crash_with())

                    # If will crash
                    if ids_will_crash_with:
                        # variable to tell where to go next (default is braking)
                        going_to_brake = True

                        # set priority
                        other_id = ids_will_crash_with[0]

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
                if self.waiting_for != 0 and self.other_task[self.waiting_for] == "PARK":
                    if self.waiting_for not in self.nearby_vehicles:
                        self.unbrake()
                        
                elif self.waiting_for != 0 and self.waiting_for_unparker:
                    if self.other_task[self.waiting_for] != "UNPARK":
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
                            or ((not self.other_is_braking[self.waiting_for] and self.other_task[self.waiting_for] != "IDLE")
                            and self.has_passed(this_id=self.waiting_for))
                            or (self.ev and self.other_task[self.waiting_for] == "IDLE") # TODO: hacky, need better law for waiting for idle (ie if idle vehicle parked, dont wait)
                            or (self.dist_from(self.waiting_for) > self.vehicle_config.braking_distance) and self.dist_from(self.waiting_for) > self.last_braking_distance):
                            should_unbrake = True
                        elif self.other_waiting_for[self.waiting_for] == self.vehicle_id: # if the vehicle you're waiting for is waiting for you
                            # you should go
                            # this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                            # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                            should_unbrake = True

                    if should_unbrake:
                        self.unbrake() # also sets brake_state to NOT_BRAKING

                if self.waiting_for != 0:
                    self.last_braking_distance = self.dist_from(self.waiting_for)

        else:
            # if reached target (pre-parking point), start parking
            self.set_ref_v(0)
            self.execute_next_task()

        if self.current_task == "IDLE":
            pass
        elif self.current_task == "PARK":
            # wait for coast to be clear, then start parking
            # everyone within range should be braking or parking or unparking

            should_go = (self.parking_maneuver is not None and self.parking_step > 0) \
                or all([
                    (self.other_task[id] not in ["UNPARK", "PARK"] and self.has_passed(other_id=id)) 
                    or (self.other_task[id] == "UNPARK" and self.other_parking_progress[id] == "")
                    or (self.other_task[id] == "PARK" and (self.other_parking_start_time[id] > self.parking_start_time or (self.should_overshoot and self.has_passed(other_id=id))))
                    or self.other_task[id] == "IDLE"
                    or self.dist_from(id) >= 2*self.vehicle_config.parking_radius for id in self.nearby_vehicles
                    ])

            if self.park_start_coords is None:
                self.park_start_coords = (self.state.x.x - self.vehicle_config.offset * np.sin(self.state.e.psi), self.state.x.y + self.vehicle_config.offset * np.cos(self.state.e.psi))
            self.update_state_parking(should_go)
        elif self.current_task == "UNPARK": # wait for coast to be clear, then start unparking
            # to start unparking, everyone within range should be (normal driving and far past us) or (waiting to unpark and spawned after us)
            # always yields to parkers in the area

            unparking_nearby_vehicles = [id for id in self.nearby_vehicles if np.abs(self.other_state[id].x.y - self.y_ref[0]) < self.vehicle_config.parking_radius]
            # old version
            # unparking_nearby_vehicles = [id for id in self.nearby_vehicles if self.dist_from(id) >= 2*self.vehicle_config.parking_radius]
            should_go = (self.unparking_maneuver is not None and self.unparking_step < len(self.unparking_maneuver.x) - 1) \
                or (all([
                    (self.other_task[id] not in ["PARK", "UNPARK"] and self.has_passed(this_id=id, parking_dist_away=7)) 
                    or self.other_task[id] == "IDLE"
                    or (self.other_task[id] == "UNPARK" and self.other_parking_start_time[id] > self.parking_start_time)
                    for id in unparking_nearby_vehicles
                    ]))
            """
            and all([self.other_parking_progress[id] != "UNPARKING" or np.linalg.norm([self.x_ref[0] - self.other_ref_pose[id].x[0], self.y_ref[0] - self.other_ref_pose[id].y[0]]) > 10 for id in self.other_vehicles]))
            """

            self.update_state_unparking(should_go)
        else: 
            self.update_state()

        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.logger.append(f't = {time}: x = {self.state.x.x:.2f}, y = {self.state.x.y:.2f}')

    def predict_intent(self, vehicle_id, history):
        """
        predict the intent of the specific vehicle
        """
        img = self.inst_centric_generator.inst_centric(vehicle_id, history)
        return self.intent_predictor.predict(img, np.array([self.state.x.x, self.state.x.y]), self.state.e.psi, self.state.v.v, 1.0)

    def get_state_dict(self):
        state_dict = {}
        state_dict['center-x'] = self.state.x.x
        state_dict['center-y'] = self.state.x.y
        state_dict['heading'] = self.state.e.psi
        state_dict['corners'] = self.vehicle_body.V
        return state_dict

    def get_other_vehicles(self):
        other_states = {i : self.other_state[i] for i in self.other_state}
        return other_states

    ##### INTENT PREDICTION #####

    def solve_intent_driving(self, time=None, timer_period=None, other_vehicles=[], history=None, coord_spot_fn=None, obstacle_corners={}):
        if self.loops <= self.loops_before_predict: 
            self.solve_classic(time=time)
        else:
            if self.loops % self.loops_between_predict == (self.loops_before_predict + 1) % self.loops_between_predict and (self.intent is None or self.intent_spot is None):
                self.solve_intent(history, coord_spot_fn)

            if self.intent_parking_step is None:
                done = self.solve_intent_control_stanley(time=time, other_vehicles=other_vehicles)

                if self.intent_spot is not None and self.intent_parking_step is None and done:
                    self.intent_parking_step = 0
                    self.intent_parking_origin = (self.state.x.x + 4, self.state.x.y - self.vehicle_config.parking_start_offset, self.state.e.psi)
                    self.current_task = "PARK"
            else:
                self.solve_parking_control(time, timer_period, P=np.diag([1, 1, 1, 1]), Q=np.diag([1, 1, 1, 1]), obstacle_corners=obstacle_corners, other_vehicles=other_vehicles)

    def solve_intent(self, history, coord_spot_fn):
        predicted_intent = self.predict_best_intent(history, coord_spot_fn)

        if predicted_intent is None:
            return
        else:
            self.intent = predicted_intent

        in_spot = coord_spot_fn(self.intent)

        if not in_spot:
            yaw = np.arctan2(self.intent[1] - self.state.x.y, self.intent[0] - self.state.x.x)
            self.intent[0] += self.vehicle_config.offset * np.sin(yaw)
            self.intent[1] -= self.vehicle_config.offset * np.cos(yaw)
        else:
            self.intent_spot = in_spot
            nearest_waypoint = self.graph.vertices[self.graph.search(self.intent)].coords
            lane_x, lane_y = nearest_waypoint[0], nearest_waypoint[1]
            # self.vehicle_config.parking_start_offset = max(-2, min(2, round((lane_y - self.state.x.y) * 4) / 4))
            self.vehicle_config.parking_start_offset = -1.75
            offset = self.vehicle_config.parking_start_offset
            dir = 'east' if self.state.x.x < self.intent[0] else 'west'
            xpos = 'left' if self.state.x.x < self.intent[0] else 'right'
            loc = 'north' if lane_y < self.intent[1] else 'south'
            # TODO: what if this doesn't give the correct maneuver (e.g. you are to the left of the spot when calculating it but will approach from the right)
            self.offset_parking_maneuver = self.offset_offline_maneuver.get_maneuver(offset=offset, driving_dir=dir, x_position=xpos, spot=loc)
            self.intent[0] += -4 if dir == 'east' else 4
            self.intent[1] = lane_y + offset

            self.change_central_occupancy(in_spot, True)

    def predict_best_intent(self, history, coord_spot_fn):
        intents = self.predict_intent(self.vehicle_id, history)
        graph = WaypointsGraph()
        graph.setup_with_vis(self.intent_extractor.vis)
        best_lanes = self.find_n_best_lanes(
            [self.state.x.x, self.state.x.y], self.state.e.psi, graph=graph, vis=self.intent_extractor.vis, predictor=self.intent_predictor)

        distributions, coordinates = self.expand_distribution(intents, best_lanes)

        top_n = list(zip(distributions, coordinates))
        top_n.sort(reverse=True)

        for _, coords in top_n:
            in_spot = coord_spot_fn(coords)

            # between 0 and 2pi
            # ang = ((np.arctan2(coords[1] - self.state.x.y, coords[0] - self.state.x.x) - self.state.e.psi) + (2*np.pi)) % (2*np.pi)
            # if ang > np.pi / 2 + np.pi / 6 and ang < 3 * np.pi / 2 - np.pi / 6: # can't have an intent behind you
            #     continue
            if in_spot and self.occupancy[in_spot]: # if occupied, can't return this
                continue
            
            return coords

        return None

    def solve_intent_control_stanley(self, time=None, other_vehicles=[]):
        """
        Having other_vehicle_objects here is just to mimic the ROS service to change values of the other vehicle. Should use this to acquire information
        """
        if self.current_task == "END":
            return True

        # Firstly update the set of nearby_vehicles
        self.update_nearby_vehicles()

        # driving control
        graph_sol = AStarPlanner(self.graph.vertices[self.graph.search([self.state.x.x, self.state.x.y])], self.graph.vertices[self.graph.search(self.intent)]).solve()
        x_ref, y_ref, yaw_ref = self.compute_ref_path(graph_sol=graph_sol)
        self.set_ref_pose(x_ref, y_ref, yaw_ref)
        if self.reached_target(target=self.intent):
            return True

        # braking controller
        if not self.is_braking:
            # normal speed controller if not braking
            if self.target_idx < self.num_waypoints() - self.vehicle_config.steps_to_end:
                self.set_ref_v(self.vehicle_config.v_cruise)
            else:
                self.set_ref_v(self.vehicle_config.v_end)

            # detect parking and unparking
            nearby_parkers = [id for id in self.nearby_vehicles if self.other_parking_progress[id] and self.other_within_parking_box(id) and not self.has_passed(other_id=id, parking_dist_away=2)]

            if nearby_parkers:
                # should only be one nearby parker, since they wait for each other
                parker_id = nearby_parkers[0]
                self.waiting_for = parker_id
                self.brake()
                self.priority = -1

                if self.other_task[parker_id] == "UNPARK":
                    self.waiting_for_unparker = True

            else: # No one is parking
                ids_will_crash_with = list(self.will_crash_with())

                # If will crash
                if ids_will_crash_with:
                    # variable to tell where to go next (default is braking)
                    going_to_brake = True

                    # set priority
                    other_id = ids_will_crash_with[0]

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
            if self.waiting_for != 0 and self.other_task[self.waiting_for] == "PARK":
                if self.waiting_for not in self.nearby_vehicles:
                    self.unbrake()
                    
            elif self.waiting_for != 0 and self.waiting_for_unparker:
                if self.other_task[self.waiting_for] != "UNPARK":
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
                        or ((not self.other_is_braking[self.waiting_for] and self.other_task[self.waiting_for] != "IDLE")
                        and self.has_passed(this_id=self.waiting_for))
                        or (self.ev and self.other_task[self.waiting_for] == "IDLE") # TODO: hacky, need better law for waiting for idle (ie if idle vehicle parked, dont wait)
                        or (self.dist_from(self.waiting_for) > self.vehicle_config.braking_distance) and self.dist_from(self.waiting_for) > self.last_braking_distance):
                        should_unbrake = True
                    elif self.other_waiting_for[self.waiting_for] == self.vehicle_id: # if the vehicle you're waiting for is waiting for you
                        # you should go
                        # this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                        # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                        should_unbrake = True

                if should_unbrake:
                    self.unbrake() # also sets brake_state to NOT_BRAKING

            if self.waiting_for != 0:
                self.last_braking_distance = self.dist_from(self.waiting_for)

        self.update_state()

        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.logger.append(f't = {time}: x = {self.state.x.x:.2f}, y = {self.state.x.y:.2f}')

        return False

    def solve_intent_control(self, time, timer_period, P=np.diag([1, 1, 0, 0]), Q=np.diag([1, 1, 0, 0]), R=np.zeros((2, 2)), obstacle_corners: Dict[Tuple, np.ndarray] = None, obstacle_As: List[np.ndarray] = None, obstacle_bs: List[np.ndarray] = None, other_vehicles = [], num_waypoints=10):
        
        cxs = [self.state.x.x + ((self.intent[0] - self.state.x.x) * i / 10) for i in range(num_waypoints)]
        cys = [self.state.x.y + ((self.intent[1] - self.state.x.y) * i / 10) for i in range(num_waypoints)]

        xref = []
        for i in range(num_waypoints):
            xref.append([cxs[i], cys[i], 0, 0])
        xref = np.array(xref)

        # solve optimal control problem
        _, feas, xOpt, uOpt, _ = self.solve_cftoc(P=P, Q=Q, R=R, N=num_waypoints, x0=np.array([self.state.x.x, self.state.x.y, self.state.v.v, self.state.e.psi]), xL=[0, 0, self.vehicle_config.v_min, None], xU=[140, 80, self.vehicle_config.v_max, None], uL=np.array([self.vehicle_config.a_min, self.vehicle_config.delta_min]), uU=np.array([self.vehicle_config.a_max, self.vehicle_config.delta_max]), xref=xref, time=time, timer_period=timer_period, obstacle_corners=obstacle_corners, obstacle_As=obstacle_As, obstacle_bs=obstacle_bs, other_vehicles=other_vehicles, initialize_with_previous=True)

        self.prediction_history[round(time * (1.0 / timer_period))] = xOpt
        self.prediction_history.pop(round(time * (1.0 / timer_period)) - 5, None) # only save 5 most recent to save space
        self.input_history[round(time * (1.0 / timer_period))] = uOpt
        self.input_history.pop(round(time * (1.0 / timer_period)) - 5, None) # only save 5 most recent to save space

        # get control (is control 0 problematic because the first xref is usually just continuing at the same speed in the same direction? not sure)
        control = uOpt[:, 0]
        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.controller.step(self.state, control[0], control[1])

        # display predictions from optimal control problem
        mpc_preds = xOpt[[0, 1]].T

        # determine if need to park
        # TODO: what if we are stopped but not close enough yet? need to get it started again somehow
        if self.intent_spot is not None and self.intent_parking_step is None and np.linalg.norm([self.state.x.x - self.intent[0], self.state.x.y - self.intent[1], self.state.v.v - 0]) < 0.5:
            self.intent_parking_step = 0
            lane_y = self.graph.vertices[self.graph.search(self.intent)].coords[1]
            self.intent_parking_origin = (self.state.x.x + 4, self.state.x.y + (self.vehicle_config.parking_start_offset if self.intent[1] < lane_y else -self.vehicle_config.parking_start_offset), self.state.e.psi)
            self.current_task = "PARK"

        return control, feas, mpc_preds
        
    def solve_parking_control(self, time, timer_period, P=np.diag([1, 1, 0, 0]), Q=np.diag([1, 1, 0, 0]), R=np.zeros((2, 2)), obstacle_corners: Dict[Tuple, np.ndarray] = None, obstacle_As: List[np.ndarray] = None, obstacle_bs: List[np.ndarray] = None, other_vehicles = [], num_waypoints=10):
        
        x_ref = self.intent_parking_origin[0] + self.offset_parking_maneuver.x[self.intent_parking_step:self.intent_parking_step + num_waypoints]
        y_ref = self.intent_parking_origin[1] + self.offset_parking_maneuver.y[self.intent_parking_step:self.intent_parking_step + num_waypoints]
        v_ref = self.offset_parking_maneuver.v[self.intent_parking_step:self.intent_parking_step + num_waypoints]
        psi_ref = self.intent_parking_origin[2] + self.offset_parking_maneuver.psi[self.intent_parking_step:self.intent_parking_step + num_waypoints]

        for _ in range(len(x_ref), num_waypoints):
            x_ref = np.append(x_ref, self.intent_parking_origin[0] + self.offset_parking_maneuver.x[-1])
            y_ref = np.append(y_ref, self.intent_parking_origin[1] + self.offset_parking_maneuver.y[-1])
            v_ref = np.append(v_ref, self.offset_parking_maneuver.v[-1])
            psi_ref = np.append(psi_ref, self.intent_parking_origin[2] + self.offset_parking_maneuver.psi[-1])

        xref = np.vstack((x_ref, y_ref, v_ref, psi_ref)).T

        # solve optimal control problem
        _, feas, xOpt, uOpt, _ = self.solve_cftoc(P=P, Q=Q, R=R, N=num_waypoints, x0=np.array([self.state.x.x, self.state.x.y, self.state.v.v, self.state.e.psi]), xL=[0, 0, self.vehicle_config.v_min, None], xU=[140, 80, self.vehicle_config.v_max, None], uL=np.array([self.vehicle_config.a_min, self.vehicle_config.delta_min]), uU=np.array([self.vehicle_config.a_max, self.vehicle_config.delta_max]), xref=xref, time=time, timer_period=timer_period, obstacle_corners=obstacle_corners, obstacle_As=obstacle_As, obstacle_bs=obstacle_bs, other_vehicles=other_vehicles)

        # get control 
        control = uOpt[:, 0]
        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.controller.step(self.state, control[0], control[1])

        # display predictions from optimal control problem
        mpc_preds = xOpt[[0, 1]].T

        self.intent_parking_step = min(self.intent_parking_step + 1, len(self.offset_parking_maneuver.x))

        # TODO: better stopping condition
        if np.linalg.norm([self.state.x.x - (self.intent_parking_origin[0] + self.offset_parking_maneuver.x[-1]), self.state.x.y - (self.intent_parking_origin[1] + self.offset_parking_maneuver.y[-1])]) < 0.6:
            self.current_task = "END"

        return control, feas, mpc_preds

        # x_ref = self.intent_parking_origin[0] + self.offset_parking_maneuver.x[self.intent_parking_step]
        # y_ref = self.intent_parking_origin[1] + self.offset_parking_maneuver.y[self.intent_parking_step]
        # v_ref = self.offset_parking_maneuver.v[self.intent_parking_step]
        # psi_ref = self.offset_parking_maneuver.psi[self.intent_parking_step]

        # self.state.x.x, self.state.x.y, self.state.v.v, self.state.e.psi = x_ref, y_ref, v_ref, psi_ref
        # self.state.t = time
        # self.state_hist.append(self.state.copy())

        # self.intent_parking_step = min(self.intent_parking_step + 1, len(self.offset_parking_maneuver.x))
        # if self.intent_parking_step == len(self.offset_parking_maneuver.x):
        #     self.current_task = "END"

    def solve_cftoc(self, P, Q, R, N, x0, xL, xU, uL, uU, xref, time, timer_period, obstacle_corners: Dict[Tuple, np.ndarray] = None, obstacle_As: List[np.ndarray] = None, obstacle_bs: List[np.ndarray] = None, other_vehicles = [], initialize_with_previous=False):
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

        # get old predictions (None if do not exist)
        model.last_time = round((time - timer_period) * (1 / timer_period))
        
        # Create state and input variables trajectory:
        if model.last_time not in self.prediction_history or not initialize_with_previous:
            model.x = pyo.Var(model.xIDX, model.tIDX)
            model.u = pyo.Var(model.uIDX, model.tIDX)
        else:
            last_pred = self.prediction_history[model.last_time]
            last_input = self.input_history[model.last_time]
            pred_data = {}
            input_data = {}
            for i in model.xIDX:
                for j in model.tIDX:
                    pred_data[(i, j)] = last_pred[i, j]
            for i in model.uIDX:
                for j in model.tIDX:
                    input_data[(i, j)] = last_input[i, j]
            model.x = pyo.Var(model.xIDX, model.tIDX, initialize=pred_data)
            model.u = pyo.Var(model.uIDX, model.tIDX, initialize=input_data)

        # Parameters
        model.static_distance = 0.01
        model.varying_distance = 0.2
        model.varying_radius = 20
    
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
        
        model.bike_const_x = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[0, t+1] == model.x[0, t] + timer_period * (model.x[2, t] * pyo.cos(model.x[3, t])) if t < N else pyo.Constraint.Skip)
        model.bike_const_y = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[1, t+1] == model.x[1, t] + timer_period * (model.x[2, t] * pyo.sin(model.x[3, t])) if t < N else pyo.Constraint.Skip)
        model.bike_const_v = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[2, t+1] == model.x[2, t] + timer_period * (model.u[0, t]) if t < N else pyo.Constraint.Skip)
        model.bike_const_psi = pyo.Constraint(model.tIDX, rule=lambda model, t: model.x[3, t+1] == model.x[3, t] + timer_period * (model.x[2, t] * pyo.tan(model.u[1, t]) / self.vehicle_body.wb) if t < N else pyo.Constraint.Skip)

        model.state_const_l = pyo.Constraint(model.xIDX, model.tIDX, rule=lambda model, i, t: model.x[i, t] <= xU[i] if xU[i] is not None else pyo.Constraint.Skip)
        model.state_const_u = pyo.Constraint(model.xIDX, model.tIDX, rule=lambda model, i, t: model.x[i, t] >= xL[i] if xL[i] is not None else pyo.Constraint.Skip)
        model.input_const_l = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] <= uU[i] if t < N else pyo.Constraint.Skip)
        model.input_const_u = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] >= uL[i] if t < N else pyo.Constraint.Skip)
        
        G = self.vehicle_body.A
        g = self.vehicle_body.b

        other_static_As = []
        other_static_bs = []

        if obstacle_corners is not None:
            for _, v in obstacle_corners.items():
                if any([np.linalg.norm([c[0] - self.state.x.x, c[1] - self.state.x.y]) < 10 for c in v]):
                    A, b = rectangle_to_polytope(v)
                    other_static_As.append(A)
                    other_static_bs.append(b)
            len(other_static_As)
        else:
            other_static_As = obstacle_As
            other_static_bs = obstacle_bs

        other_static_As = np.array(other_static_As)
        other_static_bs = np.array(other_static_bs)

        other_varying_As = []
        other_varying_bs = []

        for v in other_vehicles:
            if np.linalg.norm([model.xref[0, 0] - v.state.x.x, model.xref[1, 0] - v.state.x.y]) < model.varying_radius:
                if model.last_time in v.prediction_history:
                    other_varying_As.append([])
                    other_varying_bs.append([])
                    for w in range(N + 1):
                        last_vehicle_state = VehicleState()
                        # for last timestep, use last available prediction
                        last_vehicle_state.x.x = v.prediction_history[model.last_time][0, min(w + 1, N)]
                        last_vehicle_state.x.y = v.prediction_history[model.last_time][1, min(w + 1, N)]
                        last_vehicle_state.e.psi = v.prediction_history[model.last_time][3, min(w + 1, N)]
                        A, b = rectangle_to_polytope(get_vehicle_corners(last_vehicle_state, v.vehicle_body))
                        other_varying_As[-1].append(A)
                        other_varying_bs[-1].append(b)

        other_varying_As = np.array(other_varying_As)
        other_varying_bs = np.array(other_varying_bs)

        if len(other_static_As) > 0:

            # static collision avoidance
            model.num_others = len(other_static_As)
            model.num_halfspaces = [a.shape[0] for a in other_static_As]

            model.othervIDX = pyo.Set( initialize= range(model.num_others), ordered=True )
            model.lamIDX = pyo.Set( initialize= range(G.shape[0]), ordered=True )
            model.revLamIDX = pyo.Set( initialize= range(max(model.num_halfspaces)), ordered=True )
            model.sIDX = pyo.Set( initialize= range(G.shape[1]), ordered=True )

            model.lam = pyo.Var(model.othervIDX, model.lamIDX, model.tIDX)
            model.rev_lam = pyo.Var(model.othervIDX, model.revLamIDX, model.tIDX)
            model.s = pyo.Var(model.othervIDX, model.sIDX, model.tIDX)

            model.collision_b_const = pyo.Constraint(model.othervIDX, model.tIDX, rule=lambda model, j, t: \
                -sum( \
                    (G[l, 0] * (model.x[0, t] * pyo.cos(model.x[2, t]) + model.x[1, t] * pyo.sin(model.x[2, t])) + \
                        G[l, 1] * (-model.x[0, t] * pyo.sin(model.x[2, t]) + model.x[1, t] * pyo.cos(model.x[2, t])) + \
                            g[l] ) \
                             * model.lam[j, l, t] for l in model.lamIDX) \
                                 - sum(other_static_bs[j][l] * model.rev_lam[j, l, t] for l in model.revLamIDX if l < model.num_halfspaces[j]) >= model.static_distance)
            model.collision_Ai1_const = pyo.Constraint(model.othervIDX, model.tIDX, rule=lambda model, j, t: \
                sum( \
                    (pyo.cos(model.x[2, t]) * G[l, 0] - pyo.sin(model.x[2, t]) * G[l, 1]) \
                        * model.lam[j, l, t] for l in model.lamIDX) + model.s[j, 0, t] == 0)
            model.collision_Ai2_const = pyo.Constraint(model.othervIDX, model.tIDX, rule=lambda model, j, t: \
                sum( \
                    (pyo.sin(model.x[2, t]) * G[l, 0] + pyo.cos(model.x[2, t]) * G[l, 1]) \
                        * model.lam[j, l, t] for l in model.lamIDX) + model.s[j, 1, t] == 0)
            model.collision_Aj_const = pyo.Constraint(model.sIDX, model.othervIDX, model.tIDX, rule=lambda model, s, j, t: sum(other_static_As[j][l, s] * model.rev_lam[j, l, t] for l in model.revLamIDX if l < model.num_halfspaces[j]) - model.s[j, s, t] == 0)
            model.collision_lam1_const = pyo.Constraint(model.othervIDX, model.lamIDX, model.tIDX, rule=lambda model, j, l, t: model.lam[j, l, t] >= 0)
            model.collision_lam2_const = pyo.Constraint(model.othervIDX, model.revLamIDX, model.tIDX, rule=lambda model, j, l, t: model.rev_lam[j, l, t] >= 0)
            model.collision_s_const = pyo.Constraint(model.othervIDX, model.tIDX, rule=lambda model, j, t: sum(model.s[j, s, t] ** 2 for s in model.sIDX) <= 1)

            if len(other_varying_As) > 0:

                # varying collision avoidance (all new variables except G, g, and model.tIDX)
                model.varying_num_others = len(other_varying_As)
                model.varying_num_halfspaces = [a[0].shape[0] for a in other_varying_As]

                model.varying_othervIDX = pyo.Set( initialize= range(model.varying_num_others), ordered=True )
                model.varying_lamIDX = pyo.Set( initialize= range(G.shape[0]), ordered=True )
                model.varying_revLamIDX = pyo.Set( initialize= range(max(model.varying_num_halfspaces)), ordered=True )
                model.varying_sIDX = pyo.Set( initialize= range(G.shape[1]), ordered=True )

                model.varying_lam = pyo.Var(model.varying_othervIDX, model.varying_lamIDX, model.tIDX)
                model.varying_rev_lam = pyo.Var(model.varying_othervIDX, model.varying_revLamIDX, model.tIDX)
                model.varying_s = pyo.Var(model.varying_othervIDX, model.varying_sIDX, model.tIDX)

                model.varying_collision_b_const = pyo.Constraint(model.varying_othervIDX, model.tIDX, rule=lambda model, j, t: \
                    -sum( \
                        (G[l, 0] * (model.x[0, t] * pyo.cos(model.x[2, t]) + model.x[1, t] * pyo.sin(model.x[2, t])) + \
                            G[l, 1] * (-model.x[0, t] * pyo.sin(model.x[2, t]) + model.x[1, t] * pyo.cos(model.x[2, t])) + \
                                g[l] ) \
                                * model.varying_lam[j, l, t] for l in model.varying_lamIDX) \
                                    - sum(other_varying_bs[j, t][l] * model.varying_rev_lam[j, l, t] for l in model.varying_revLamIDX if l < model.varying_num_halfspaces[j]) >= model.varying_distance)
                model.varying_collision_Ai1_const = pyo.Constraint(model.varying_othervIDX, model.tIDX, rule=lambda model, j, t: \
                    sum( \
                        (pyo.cos(model.x[2, t]) * G[l, 0] - pyo.sin(model.x[2, t]) * G[l, 1]) \
                            * model.varying_lam[j, l, t] for l in model.varying_lamIDX) + model.varying_s[j, 0, t] == 0)
                model.varying_collision_Ai2_const = pyo.Constraint(model.varying_othervIDX, model.tIDX, rule=lambda model, j, t: \
                    sum( \
                        (pyo.sin(model.x[2, t]) * G[l, 0] + pyo.cos(model.x[2, t]) * G[l, 1]) \
                            * model.varying_lam[j, l, t] for l in model.varying_lamIDX) + model.varying_s[j, 1, t] == 0)
                model.varying_collision_Aj_const = pyo.Constraint(model.varying_sIDX, model.varying_othervIDX, model.tIDX, rule=lambda model, s, j, t: sum(other_varying_As[j, t][l, s] * model.varying_rev_lam[j, l, t] for l in model.varying_revLamIDX if l < model.varying_num_halfspaces[j]) - model.varying_s[j, s, t] == 0)
                model.varying_collision_lam1_const = pyo.Constraint(model.varying_othervIDX, model.varying_lamIDX, model.tIDX, rule=lambda model, j, l, t: model.varying_lam[j, l, t] >= 0)
                model.varying_collision_lam2_const = pyo.Constraint(model.varying_othervIDX, model.varying_revLamIDX, model.tIDX, rule=lambda model, j, l, t: model.varying_rev_lam[j, l, t] >= 0)
                model.varying_collision_s_const = pyo.Constraint(model.varying_othervIDX, model.tIDX, rule=lambda model, j, t: sum(model.varying_s[j, s, t] ** 2 for s in model.varying_sIDX) <= 1)

        results = self.solver.solve(model)
        
        if str(results.solver.termination_condition) == "optimal":
            feas = True
        else:
            feas = False
                
        xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
        uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T
        
        JOpt = model.cost()
        
        return [model, feas, xOpt, uOpt, JOpt]

    
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
        for i, lane in enumerate(all_lanes):
            astar_dist, astar_dir = predictor.compute_Astar_dist_dir(
                current_state, lane.coords, global_heading)
            heapq.heappush(lanes, (-astar_dir, astar_dist, i, lane.coords)) # i is to avoid issues when two heap elements are the same

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
            _, _, _, coords = heapq.heappop(lanes)
            coordinates.append(coords)
            distributions.append(p_minus * scales[i])

        return distributions, coordinates

    # stride was 10 on dlp example (with 25 FPS, so sampling every 0.4 seconds)
    def get_data_for_instance(self, extractor: TransformerDataProcessor, global_intent_pose: np.array, history: np.ndarray, stride: int=4, time_history: int=10, future: int=10, img_size: int=100) -> Tuple[np.array, np.array, np.array]:
        """
        returns image, trajectory_history, and trajectory future for given instance
        """
        img_transform=transforms.ToTensor()
        image_feature = self.inst_centric_generator.inst_centric(self.vehicle_id, history)

        image_feature = self.label_target_spot(self, image_feature)

        curr_pose = np.array([self.state.x.x,
                            self.state.x.y, self.state.e.psi])
        rot = np.array([[np.cos(-curr_pose[2]), -np.sin(-curr_pose[2])], [np.sin(-curr_pose[2]), np.cos(-curr_pose[2])]])
        
        local_intent_coords = np.dot(rot, global_intent_pose[:2]-curr_pose[:2])
        local_intent_pose = np.expand_dims(local_intent_coords, axis=0)

        # determine start index to gather history from
        start_idx = -1
        while start_idx - stride > -stride * time_history and start_idx - stride >= -len(self.state_hist):
            start_idx -= stride

        image_history = []
        trajectory_history = []
        for i in range(start_idx, 0, stride):
            state = self.state_hist[i]
            pos = np.array([state.x.x, state.x.y])
            translated_pos = np.dot(rot, pos-curr_pose[:2])
            trajectory_history.append(Tensor(
                [translated_pos[0], translated_pos[1], state.e.psi - curr_pose[2]]))

            image_feature = self.inst_centric_generator.inst_centric(self.vehicle_id, history)
            image_feature = self.label_target_spot(self, image_feature, curr_pose)
            
            image_tensor = img_transform(image_feature.resize((img_size, img_size)))
            image_history.append(image_tensor)
        
        return torch.stack(image_history)[None], torch.stack(trajectory_history)[None], torch.from_numpy(local_intent_pose)[None]

    def label_target_spot(self, inst_centric_view: np.array, center_pose: np.ndarray=None, r=1.25) -> np.array:
        """
        Returns image frame with target spot labeled
        center_pose: If None, the inst_centric_view is assumed to be around the current instance. If a numpy array (x, y, heading) is given, it is the specified center.
        """
        all_spots = self.spot_detector.detect(inst_centric_view)

        if center_pose is None:
            current_state = np.array([self.state.x.x, self.state.x.y, self.state.e.psi])
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