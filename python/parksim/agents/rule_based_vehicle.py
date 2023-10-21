from typing import Dict, List, Set, Tuple
from pathlib import Path
import numpy as np
import pickle
import time
import array
from collections import deque

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController
from parksim.controller.mpc_controller import MPC

from parksim.vehicle_types import VehicleBody, VehicleConfig, VehicleInfo, VehicleTask
from parksim.pytypes import VehiclePrediction, VehicleState

from parksim.route_planner.a_star import AStarGraph, AStarPlanner
from parksim.route_planner.graph import Vertex, WaypointsGraph
from parksim.utils.get_corners import get_vehicle_corners
from parksim.utils.interpolation import interpolate_states_inputs
from parksim.path_planner.offline_maneuver import OfflineManeuver

from parksim.intent_predict.cnn.predictor import Predictor
from parksim.intent_predict.cnn.visualizer.instance_centric_generator import (
    InstanceCentricGenerator,
)
from parksim.intent_predict.cnn.intent_sampler import IntentSampler
from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from parksim.spot_detector.detector import LocalDetector


class RuleBasedVehicle(AbstractAgent):
    def __init__(
        self,
        vehicle_id: int,
        vehicle_body: VehicleBody,
        vehicle_config: VehicleConfig,
        controller: StanleyController = StanleyController(),
        motion_predictor: StanleyController = StanleyController(),
        inst_centric_generator=InstanceCentricGenerator(),
        intent_predictor=Predictor(),
        intent_extractor: CNNDataProcessor = None,
        spot_detector: LocalDetector = None,
        electric_vehicle: bool = False,
        intent_vehicle: bool = False,
    ):
        self.vehicle_id = vehicle_id
        self.loops = 0

        # State and Reference Waypoints
        self.state: VehicleState = VehicleState()  # state
        self.info: VehicleInfo = VehicleInfo()  # Info
        self.disp_text: str = str(self.vehicle_id)

        self.state_hist: List[VehicleState] = []  # State history

        self.x_ref = []  # x coordinates for waypoints
        self.y_ref = []  # y coordinates for waypoints
        self.yaw_ref = []  # yaws for waypoints
        self.v_ref = 0  # target speed

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
        self.mpc = MPC()
        self.mpc.setup()
        self.motion_predictor = motion_predictor
        self.intent_predictor = intent_predictor  # cnnV2

        self.target_idx = 0

        self.graph: WaypointsGraph = None
        self.entrance_vertex: int = None

        # intent predict
        self.intent_vehicle = intent_vehicle

        if self.intent_vehicle:
            self.intent_sampler = IntentSampler(
                inst_centric_generator,
                intent_predictor,
                intent_extractor,
                spot_detector,
            )
            self.intent = None

            self.intent_maneuver_params = None
            self.intent_spot = None

            self.loops_before_predict = 5
            self.loops_between_predict = 5

        # parking stuff
        self.occupancy = None
        self.parking_spaces = None
        self.north_spot_idx_ranges: List[Tuple[int, int]] = None
        self.spot_y_offset: float = None

        self.spot_index = None
        self.should_overshoot = False  # overshooting or undershooting the spot?
        self.park_start_coords = None

        self.offline_maneuver: OfflineManeuver = None
        self.overshoot_ranges: Dict[str, List[Tuple[int]]] = None

        self.parking_start_time = float(
            "inf"
        )  # inf means haven't start parking or unparking. Anything above 0 is parking

        self.parking_maneuver = None
        self.parking_step = (
            0 if not self.intent_vehicle else -1
        )  # parking handled slightly differently for intent vehicle

        # unparking stuff
        self.unparking_maneuver = None
        self.unparking_step = -1
        self.unparking_x_ref = None
        self.unparking_y_ref = None
        self.unparking_yaw_ref = None

        # braking stuff
        self.is_braking = False  # are we braking?
        self.last_braking_distance = None  # to prevent braking because will crash, then immediate unbraking because not inside braking distance
        self._pre_brake_target_speed = 0  # speed to restore when unbraking
        self.priority = 0  # priority for going after braking
        self.waiting_for: int = 0  # vehicle waiting for before we go. We start indexing vehicles from 1, so 0 means no vehicle
        self.waiting_for_unparker = (
            False  # need special handling for waiting for unparker
        )

        # ev charging
        self.charging_spots = [39, 40, 41]  # TODO: put this in a yaml
        self.ev = electric_vehicle
        self.ev_charging_state = None  # none = not ev, 0 = waiting to charge, 1 = charging, 2 = done charging TODO: make enum

        self.logger = deque(maxlen=100)

        # ============= Information of other vehicles ===========
        self.other_vehicles: Set(int) = set()  # Other vehicle ids
        self.nearby_vehicles: Set(int) = set()  # Nearby vehicles that we are interested
        self.other_state: Dict[int, VehicleState] = {}
        self.other_ref_pose: Dict[int, VehiclePrediction] = {}
        self.other_ref_v: Dict[int, float] = {}
        self.other_target_idx: Dict[int, int] = {}
        self.other_priority: Dict[int, int] = {}
        self.other_task: Dict[int, str] = {}  # The current task of other vehicle
        self.other_parking_progress: Dict[
            int, str
        ] = (
            {}
        )  # Other vehicles will broadcast "PARKING" if vehicle.is_parking(), "UNPARKING" if vehicle.is_unparking(), None otherwise
        self.other_parking_start_time: Dict[int, float] = {}
        self.other_is_braking: Dict[int, str] = {}
        self.other_waiting_for: Dict[int, int] = {}
        self.other_is_all_done: Dict[int, bool] = {}

        # ============== Method to exchange information
        self.method_to_change_central_occupancy = None

    def set_ref_pose(
        self, x_ref: List[float], y_ref: List[float], yaw_ref: List[float]
    ):
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.yaw_ref = yaw_ref

        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        self.target_idx = self.controller.calc_target_index(self.state)[
            0
        ]  # waypoint the vehicle is targeting

    def set_ref_v(self, v_ref: float):
        self.v_ref = v_ref

    def set_target_idx(self, target_idx: int):
        self.target_idx = target_idx

    def set_vehicle_state(
        self, state: VehicleState = None, spot_index: int = None, heading: float = None
    ):
        if state is not None:
            self.state = state
        elif spot_index is not None:
            assert (
                self.parking_spaces is not None
            ), "Please run load_parking_spaces first."

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
                if (
                    task.name == "PARK"
                    and task.target_spot_index in self.charging_spots
                ):  # will park in charging spot
                    self.ev_charging_state = 0
            if self.ev_charging_state is None:  # already done
                self.ev_charging_state = 2

    def load_parking_spaces(self, spots_data_path: str):
        home_path = str(Path.home())
        with open(home_path + spots_data_path, "rb") as f:
            data = pickle.load(f)
            self.parking_spaces = data["parking_spaces"]
            self.overshoot_ranges = data["overshoot_ranges"]
            self.north_spot_idx_ranges = data["north_spot_idx_ranges"]
            self.spot_y_offset = data["spot_y_offset"]

    def load_graph(self, waypoints_graph_path: str):
        """
        waypoints_graph_path: path to WaypointGraph object pickle
        entrance_coords: The (x,y) coordinates of the entrance
        """
        home_path = str(Path.home())
        with open(home_path + waypoints_graph_path, "rb") as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            entrance_coords = data["entrance_coords"]

        # Default entrance vertex
        self.entrance_vertex = self.graph.search(entrance_coords)

    def load_maneuver(self, offline_maneuver_path: str):
        home_path = str(Path.home())
        self.offline_maneuver = OfflineManeuver(
            pickle_file=home_path + offline_maneuver_path
        )

    def load_intent_model(self, model_path: str):
        """
        load_graph must be called before load_intent_model.
        """
        home_path = str(Path.home())
        self.intent_predictor.load_model(
            waypoints=self.graph, model_path=home_path + model_path
        )

    def compute_ref_path(
        self, graph_sol: AStarGraph, offset: float = None, spot_index: int = None
    ):
        if not offset:
            offset = self.vehicle_config.offset

        if spot_index is None:
            # exiting
            if len(graph_sol.vertices) == 0:  # just point right by default
                shifted_coords = np.array(
                    [
                        self.state.x.x - np.sin(self.state.e.psi) * offset,
                        self.state.x.y + np.cos(self.state.e.psi) * offset,
                    ]
                )
                end_coords = np.array([self.state.x.x, self.state.x.y])
                end_vertex_idx = self.graph.search(end_coords)
                graph_sol.vertices = [
                    Vertex(shifted_coords),
                    self.graph.vertices[end_vertex_idx],
                ]

            x_ref, y_ref, yaw_ref = graph_sol.compute_ref_path(offset)
        else:
            if len(graph_sol.vertices) == 0:  # just point right by default
                start_coords = np.array([self.state.x.x, self.state.x.y])
                start_vertex_idx = self.graph.search(start_coords)
                last_x, last_y = self.graph.vertices[start_vertex_idx].coords
                graph_sol.vertices = [
                    Vertex(np.array([last_x, last_y])),
                    Vertex(np.array([last_x + 4, last_y])),
                ]

            elif len(graph_sol.vertices) == 1:  # just point right by default
                last_x, last_y = graph_sol.vertices[0].coords
                graph_sol.vertices.append([Vertex(np.array([last_x + 4, last_y]))])

            else:
                # parking
                last_edge = graph_sol.edges[-1]
                pointed_right = last_edge.v2.coords[0] - last_edge.v1.coords[0] > 0

                if pointed_right:
                    overshoot_ranges = self.overshoot_ranges["pointed_right"]
                else:
                    overshoot_ranges = self.overshoot_ranges["pointed_left"]

                self.should_overshoot = any(
                    [
                        spot_index >= r[0] and spot_index <= r[1]
                        for r in overshoot_ranges
                    ]
                )

                last_x, last_y = last_edge.v2.coords

                if self.should_overshoot:
                    # add point past the final waypoint, that signifies going past the spot by 4 meters, so it parks in the right place
                    # if the last edge was pointed right, offset to the right
                    if pointed_right:
                        new_vertex = Vertex(np.array([last_x + 4, last_y]))
                    else:
                        new_vertex = Vertex(np.array([last_x - 4, last_y]))
                    graph_sol.vertices.append(new_vertex)
                else:
                    # if the last edge was pointed right, offset to the left
                    if pointed_right:
                        new_vertex = Vertex(np.array([last_x - 4, last_y]))
                    else:
                        new_vertex = Vertex(np.array([last_x + 4, last_y]))

                    last_waypoint = len(graph_sol.vertices) - 1
                    for i, v in enumerate(reversed(graph_sol.vertices)):
                        if pointed_right:
                            if v.coords[0] < new_vertex.coords[0]:
                                last_waypoint = -i - 1
                                break
                        else:
                            if v.coords[0] > new_vertex.coords[0]:
                                last_waypoint = -i - 1
                                break

                    graph_sol.vertices = graph_sol.vertices[: last_waypoint + 1]
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

            is_north_spot = any(
                [
                    abs(task.target_spot_index) >= r[0]
                    and abs(task.target_spot_index) <= r[1]
                    for r in self.north_spot_idx_ranges
                ]
            )
            y_offset = -self.spot_y_offset if is_north_spot else self.spot_y_offset
            waypoint_coords = [
                self.parking_spaces[abs(task.target_spot_index)][0],
                self.parking_spaces[abs(task.target_spot_index)][1] + y_offset,
            ]

            graph_sol = AStarPlanner(
                self.graph.vertices[start_vertex_idx],
                self.graph.vertices[self.graph.search(waypoint_coords)],
            ).solve()

            x_ref, y_ref, yaw_ref = self.compute_ref_path(
                graph_sol=graph_sol, spot_index=task.target_spot_index
            )

        elif task.target_coords is not None:
            # Travel to a coordinates
            graph_sol = AStarPlanner(
                self.graph.vertices[start_vertex_idx],
                self.graph.vertices[self.graph.search(task.target_coords)],
            ).solve()

            if len(graph_sol.edges) == 0:  # just go to a waypoint
                x_ref = [self.state.x.x, task.target_coords[0]]
                y_ref = [self.state.x.y, task.target_coords[1]]
                yaw_ref = [self.state.e.psi, self.state.e.psi]
            else:  # compute a-star path
                x_ref, y_ref, yaw_ref = self.compute_ref_path(
                    graph_sol=graph_sol, spot_index=None
                )

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
                    (
                        self.unparking_x_ref,
                        self.unparking_y_ref,
                        self.unparking_yaw_ref,
                    ) = (self.x_ref, self.y_ref, self.yaw_ref)
                else:
                    raise ValueError(
                        "UNPARK task should be followed with a CRUISE task."
                    )

                if self.ev and task.target_spot_index in self.charging_spots:
                    self.ev_charging_state = 2

            elif task.name == "IDLE":
                if task.duration is not None:
                    self.idle_duration = task.duration
                else:
                    self.idle_end_time = task.end_time

                # ev charging
                # if charging, set variable
                if (
                    self.ev
                    and len(self.task_profile) > 0
                    and self.task_profile[0].name == "UNPARK"
                    and self.task_profile[0].target_spot_index in self.charging_spots
                ):
                    self.ev_charging_state = 1
            else:
                raise ValueError(f"Undefined task name. {task.name} is received.")

            self.task_history.append(task)
        else:
            # Finished all tasks
            self.current_task = "END"

    def reached_target(self, target=None):
        if target is None:
            target = [self.x_ref[-1], self.y_ref[-1]]
        dist = np.linalg.norm([self.state.x.x - target[0], self.state.x.y - target[1]])
        ang = (
            (
                np.arctan2(target[1] - self.state.x.y, target[0] - self.state.x.x)
                - self.state.e.psi
            )
            + (2 * np.pi)
        ) % (2 * np.pi)
        reached_tgt = (
            dist < self.vehicle_config.braking_distance / 2
            and ang > (np.pi / 2)
            and ang < (3 * np.pi / 2)
        )
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
        self.info.ref_pose.x = array.array("d", self.x_ref)
        self.info.ref_pose.y = array.array("d", self.y_ref)
        self.info.ref_pose.psi = array.array("d", self.yaw_ref)
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
        return np.linalg.norm(
            [
                self.other_state[other_id].x.x - self.state.x.x,
                self.other_state[other_id].x.y - self.state.x.y,
            ]
        )

    def will_crash_with(self) -> Set[int]:
        will_crash_with = set()

        # create states for looking ahead
        look_ahead_state = self.state.copy()
        other_look_ahead_states = [
            self.other_state[id].copy() for id in self.nearby_vehicles
        ]

        # for each time step, looking ahead
        for _ in range(self.vehicle_config.look_ahead_timesteps):
            # calculate new positions
            self.motion_predictor.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
            self.motion_predictor.set_ref_v(self.v_ref)
            self.motion_predictor.set_target_idx(self.target_idx)
            ai, di, _ = self.motion_predictor.solve(look_ahead_state, self.is_braking)
            self.motion_predictor.step(look_ahead_state, ai, di)

            for id, other_look_ahead_state in zip(
                self.nearby_vehicles, other_look_ahead_states
            ):
                if id not in will_crash_with:  # for efficiency
                    if not self.other_ref_pose[id].x:
                        continue

                    self.motion_predictor.set_ref_pose(
                        self.other_ref_pose[id].x,
                        self.other_ref_pose[id].y,
                        self.other_ref_pose[id].psi,
                    )
                    self.motion_predictor.set_ref_v(self.other_ref_v[id])
                    self.motion_predictor.set_target_idx(self.other_target_idx[id])
                    ai, di, _ = self.motion_predictor.solve(
                        other_look_ahead_state, self.other_is_braking[id]
                    )
                    self.motion_predictor.step(other_look_ahead_state, ai, di)

            # detect crash
            for id, other_look_ahead_state in zip(
                self.nearby_vehicles, other_look_ahead_states
            ):
                if id not in will_crash_with:  # for efficiency
                    if self.will_collide(
                        look_ahead_state, other_look_ahead_state, self.vehicle_body
                    ):
                        # NOTE: Here we assume all other vehicles have the same vehicle body as us
                        will_crash_with.add(id)

        return will_crash_with

    def should_start_parking(self):
        return all(
            [
                (
                    self.other_task[id] not in ["UNPARK", "PARK"]
                    and self.has_passed(other_id=id)
                )
                or (
                    self.other_task[id] == "UNPARK"
                    and self.other_parking_progress[id] == ""
                )
                or (
                    self.other_task[id] == "PARK"
                    and (
                        self.other_parking_start_time[id] > self.parking_start_time
                        or (self.should_overshoot and self.has_passed(other_id=id))
                    )
                )
                or self.other_task[id] == "IDLE"
                or self.dist_from(id) >= 2 * self.vehicle_config.parking_radius
                or (
                    self.intent_vehicle
                    and self.intent_spot in [19, 20, 21, 22, 23]
                    and self.other_is_braking[id]
                    and self.other_state[id].x.y < 60
                )  # edge case for intent
                for id in self.nearby_vehicles
            ]
        )

    def should_go_before(self, other_id):
        """
        Determines if one car should go before another. Does it based on angles: if one vehicle has gone more past the other vehicle than the other, it should go first.
        """
        this_ang = (
            (
                np.arctan2(
                    self.other_state[other_id].x.y - self.state.x.y,
                    self.other_state[other_id].x.x - self.state.x.x,
                )
                - self.state.e.psi
            )
            + (2 * np.pi)
        ) % (2 * np.pi)
        other_ang = (
            (
                np.arctan2(
                    self.state.x.y - self.other_state[other_id].x.y,
                    self.state.x.x - self.other_state[other_id].x.x,
                )
                - self.other_state[other_id].e.psi
            )
            + (2 * np.pi)
        ) % (2 * np.pi)
        this_ang_centered = this_ang if this_ang < np.pi else this_ang - 2 * np.pi
        other_ang_centered = other_ang if other_ang < np.pi else other_ang - 2 * np.pi
        return abs(this_ang_centered) > abs(other_ang_centered)

    def has_passed(
        self, this_id: int = None, other_id: int = None, parking_dist_away=None
    ):
        """
        If the rear corners of this vehicle have passed the front corners of the other vehicle, we say this vehicle has passed the other vehicle.
        parking_dist_away: additional check, if this_id's x-coordinate is parking_dist_away past other_id's x-coordinate
        """
        if this_id is None or this_id == self.vehicle_id:
            this_corners = self.get_corners()
            this_state = self.state
            this_psi = (this_state.e.psi + (2 * np.pi)) % (2 * np.pi)
        else:
            this_corners = self.get_corners(self.other_state[this_id])
            this_state = self.other_state[this_id]
            this_psi = (this_state.e.psi + (2 * np.pi)) % (2 * np.pi)

        if other_id is None or other_id == self.vehicle_id:
            other_corners = self.get_corners()
            other_state = self.state
        else:
            other_corners = self.get_corners(
                self.other_state[other_id]
            )  # NOTE: For now, assume the other vehicle has the same vehicle body
            other_state = self.other_state[other_id]

        for this_corner in [this_corners[0], this_corners[1]]:
            for other_corner in [other_corners[2], other_corners[3]]:
                ang = (
                    (
                        np.arctan2(
                            other_corner[1] - this_corner[1],
                            other_corner[0] - this_corner[0],
                        )
                        - this_psi
                    )
                    + (2 * np.pi)
                ) % (2 * np.pi)
                if ang < (np.pi / 2) or ang > (3 * np.pi) / 2:
                    return False
        if parking_dist_away is not None:
            if this_psi > np.pi / 2 and this_psi < np.pi * 3 / 2:  # facing west
                if this_state.x.x - other_state.x.x > -parking_dist_away:
                    return False
            else:
                if this_state.x.x - other_state.x.x < parking_dist_away:
                    return False
        return True

    def other_within_parking_box(self, other_id):
        ang = (
            (
                np.arctan2(
                    self.other_state[other_id].x.y - self.state.x.y,
                    self.other_state[other_id].x.x - self.state.x.x,
                )
                - self.state.e.psi
            )
            + (2 * np.pi)
        ) % (2 * np.pi)
        dist = self.dist_from(other_id)
        if (
            ang < self.vehicle_config.parking_ahead_angle
            or ang > 2 * np.pi - self.vehicle_config.parking_ahead_angle
        ):
            return dist < 2 * self.vehicle_config.parking_radius
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

    def solve_parking_control_teleport(self, time, advance=True):
        if self.parking_maneuver is None:  # start parking
            # get parking parameters
            direction = (
                "west"
                if self.state.e.psi > np.pi / 2 or self.state.e.psi < -np.pi / 2
                else "east"
            )
            if self.should_overshoot:
                location = (
                    "right" if (direction == "east") else "left"
                )  # we are designed to overshoot the spot
            else:
                location = (
                    "left" if (direction == "east") else "right"
                )  # we are designed to undershoot the spot
            pointing = (
                "up" if np.random.rand() < 0.5 else "down"
            )  # random for diversity
            spot = (
                "north"
                if any(
                    [
                        self.spot_index >= r[0] and self.spot_index <= r[1]
                        for r in self.north_spot_idx_ranges
                    ]
                )
                else "south"
            )

            # get parking maneuver
            offline_maneuver = self.offline_maneuver.get_maneuver(
                [
                    self.park_start_coords[0] - 4
                    if location == "right"
                    else self.park_start_coords[0] + 4,
                    self.park_start_coords[1],
                ],
                direction,
                location,
                spot,
                pointing,
            )

            time_seq = np.arange(
                start=offline_maneuver.t[0],
                stop=offline_maneuver.t[-1],
                step=self.controller.dt,
            )

            self.parking_maneuver = interpolate_states_inputs(
                offline_maneuver, time_seq
            )

            self.set_parking_start_time(time)

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

    def solve_unparking_control_teleport(self, time, advance=True):
        if self.unparking_maneuver is None:  # start unparking
            # get unparking parameters
            direction = (
                "west"
                if self.x_ref[0] >= self.x_ref[1]
                and self.spot_index not in [91, 183, 275]
                else "east"
            )  # if first direction of travel is left, face west
            location = (
                "right" if np.random.rand() < 0.5 else "left"
            )  # random for diversity
            pointing = "up" if self.state.e.psi > 0 else "down"  # determine from state
            spot = (
                "north"
                if any(
                    [
                        abs(self.spot_index) >= r[0] and abs(self.spot_index) <= r[1]
                        for r in self.north_spot_idx_ranges
                    ]
                )
                else "south"
            )

            # get parking maneuver
            offline_maneuver = self.offline_maneuver.get_maneuver(
                [
                    self.state.x.x if location == "right" else self.state.x.x,
                    self.state.x.y - 6.25 if spot == "north" else self.state.x.y + 6.25,
                ],
                direction,
                location,
                spot,
                pointing,
            )

            time_seq = np.arange(
                start=offline_maneuver.t[0],
                stop=offline_maneuver.t[-1],
                step=self.controller.dt,
            )

            self.unparking_maneuver = interpolate_states_inputs(
                offline_maneuver, time_seq
            )

            # set initial unparking state
            self.unparking_step = len(self.unparking_maneuver.x) - 1

            self.set_parking_start_time(time)

        # get step
        step = self.unparking_step

        # set state
        self.state.x.x = self.unparking_maneuver.x[step]
        self.state.x.y = self.unparking_maneuver.y[step]
        self.state.e.psi = self.unparking_maneuver.psi[step]
        self.state.v.v = self.unparking_maneuver.v[step]

        self.state.u.u_a = self.unparking_maneuver.u_a[step]
        self.state.u.u_steer = self.unparking_maneuver.u_steer[step]

        if self.unparking_step == 0:  # done unparking
            self.change_central_occupancy(self.spot_index, False)

            self.reset_parking_related()
            self.execute_next_task()
        else:
            # update parking step if advancing
            self.unparking_step -= 1 if advance else 0

    def reset_parking_related(self):
        # inf means haven't start parking or unparking. Anything above 0 is parking
        self.parking_start_time = float("inf")

        self.parking_maneuver = None
        self.parking_step = 0

        # unparking stuff
        self.unparking_maneuver = None
        self.unparking_step = -1

        self.park_start_coords = None

    def set_parking_start_time(self, time):
        """
        Sets self.parking_start_time, with a tiebreaker so two vehicles can't have the same time
        """
        self.parking_start_time = time + self.vehicle_id * 0.0000001

    def get_corners(self, state: VehicleState = None, vehicle_body: VehicleBody = None):
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
        return (
            self.current_task == "PARK"
            and self.parking_maneuver is not None
            and self.parking_step > (0 if not self.intent_vehicle else -1)
            and self.parking_step < len(self.parking_maneuver.x) - 1
        )

    def is_unparking(self):
        return (
            self.current_task == "UNPARK"
            and self.unparking_maneuver is not None
            and self.unparking_step < len(self.unparking_maneuver.x) - 1
            and self.unparking_step > 0
        ) and not (self.current_task == "UNPARK" and self.unparking_step == -1)

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

    def solve(
        self,
        time=None,
        timer_period=None,
        other_vehicles=[],
        history=None,
        coord_spot_fn=None,
        obstacle_corners={},
        intent_use_obstacles=True,
    ):
        """
        Update the vehicle state
        """
        if self.intent_vehicle:
            # drive according to intent prediction
            mpc_preds = self.solve_intent_based_control(
                time=time,
                timer_period=timer_period,
                other_vehicles=other_vehicles,
                history=history,
                coord_spot_fn=coord_spot_fn,
                obstacle_corners=obstacle_corners,
                intent_use_obstacles=intent_use_obstacles,
            )
            return mpc_preds
        else:
            # drive according to task profile
            self.solve_task_based_control(time, coord_spot_fn=coord_spot_fn)
            return None

    def solve_task_based_control(self, time=None, coord_spot_fn=None):
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

            if (
                self.idle_duration is not None
                and time - self.idle_start_time >= self.idle_duration
                or self.idle_end_time is not None
                and time >= self.idle_end_time
            ):
                self.idle_start_time = None
                self.idle_duration = None
                self.idle_end_time = None
                self.execute_next_task()
        elif not self.reached_target():
            # normal driving (haven't reached pre-parking point)
            self.standard_driving_control(coord_spot_fn)
        else:
            # if reached target (pre-parking point), start parking
            self.set_ref_v(0)
            self.execute_next_task()

        if self.current_task == "IDLE":
            pass
        elif self.current_task == "PARK":
            # wait for coast to be clear, then start parking
            # everyone within range should be braking or parking or unparking

            should_go = (
                self.parking_maneuver is not None and self.parking_step > 0
            ) or self.should_start_parking()

            if self.park_start_coords is None:
                self.park_start_coords = (
                    self.state.x.x
                    - self.vehicle_config.offset * np.sin(self.state.e.psi),
                    self.state.x.y
                    + self.vehicle_config.offset * np.cos(self.state.e.psi),
                )
            self.solve_parking_control_teleport(time, should_go)
        elif (
            self.current_task == "UNPARK"
        ):  # wait for coast to be clear, then start unparking
            # to start unparking, everyone within range should be (normal driving and far past us) or (waiting to unpark and spawned after us)
            # always yields to parkers in the area

            unparking_nearby_vehicles = [
                id
                for id in self.nearby_vehicles
                if np.abs(self.other_state[id].x.y - self.y_ref[0])
                < self.vehicle_config.parking_radius
            ]
            # old version
            # unparking_nearby_vehicles = [id for id in self.nearby_vehicles if self.dist_from(id) >= 2*self.vehicle_config.parking_radius]
            should_go = (
                self.unparking_maneuver is not None
                and self.unparking_step < len(self.unparking_maneuver.x) - 1
            ) or (
                all(
                    [
                        (
                            self.other_task[id] not in ["PARK", "UNPARK"]
                            and self.has_passed(this_id=id, parking_dist_away=7)
                        )
                        or self.other_task[id] == "IDLE"
                        or (
                            self.other_task[id] == "UNPARK"
                            and self.other_parking_start_time[id]
                            > self.parking_start_time
                        )
                        for id in unparking_nearby_vehicles
                    ]
                )
            )
            """
            and all([self.other_parking_progress[id] != "UNPARKING" or np.linalg.norm([self.x_ref[0] - self.other_ref_pose[id].x[0], self.y_ref[0] - self.other_ref_pose[id].y[0]]) > 10 for id in self.other_vehicles]))
            """

            self.solve_unparking_control_teleport(time, should_go)
        else:
            self.update_state()

        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.logger.append(
            f"t = {time}: x = {self.state.x.x:.2f}, y = {self.state.x.y:.2f}"
        )

    def standard_driving_control(self, coord_spot_fn):
        # braking controller
        if not self.is_braking:
            # normal speed controller if not braking
            if (
                self.target_idx
                < self.num_waypoints() - self.vehicle_config.steps_to_end
            ):
                self.set_ref_v(self.vehicle_config.v_cruise)
            else:
                self.set_ref_v(self.vehicle_config.v_end)

            # detect parking and unparking
            nearby_parkers = [
                id
                for id in self.nearby_vehicles
                if self.other_parking_progress[id]
                and self.other_within_parking_box(id)
                and not self.has_passed(other_id=id, parking_dist_away=2)
            ]

            if nearby_parkers:
                # should only be one nearby parker, since they wait for each other
                parker_id = nearby_parkers[0]
                self.waiting_for = parker_id
                self.brake()
                self.priority = -1

                if self.other_task[parker_id] == "UNPARK":
                    self.waiting_for_unparker = True

            else:  # No one is parking
                ids_will_crash_with = list(self.will_crash_with())

                # If will crash
                if ids_will_crash_with:
                    # variable to tell where to go next (default is braking)
                    going_to_brake = True

                    # set priority
                    other_id = ids_will_crash_with[0]

                    if not any(
                        [self.should_go_before(id) for id in ids_will_crash_with]
                    ):
                        going_to_brake = True  # go straight to waiting, no priority calculations necessary

                        self.priority = self.other_priority[other_id] - 1

                        self.waiting_for = other_id
                    else:  # leading car
                        going_to_brake = False  # don't brake

                    if going_to_brake:
                        self.brake()

        else:  # waiting / braking
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
                    if (
                        self.waiting_for not in self.nearby_vehicles
                        or (
                            (
                                not self.other_is_braking[self.waiting_for]
                                and self.other_task[self.waiting_for] != "IDLE"
                            )
                            and self.has_passed(this_id=self.waiting_for)
                        )
                        or (
                            self.ev and self.other_task[self.waiting_for] == "IDLE"
                        )  # TODO: hacky, need better law for waiting for idle (ie if idle vehicle parked, dont wait)
                        or (
                            (
                                self.dist_from(self.waiting_for)
                                > self.vehicle_config.braking_distance
                            )
                            and self.dist_from(self.waiting_for)
                            > self.last_braking_distance
                        )
                        or (
                            self.other_task[self.waiting_for] == "IDLE"
                            and self.other_state[self.waiting_for]
                            and coord_spot_fn(
                                [
                                    self.other_state[self.waiting_for].x.x,
                                    self.other_state[self.waiting_for].x.y,
                                ]
                            )
                            is not None
                        )
                    ):
                        should_unbrake = True
                    elif (
                        self.other_waiting_for[self.waiting_for] == self.vehicle_id
                    ):  # if the vehicle you're waiting for is waiting for you
                        # you should go
                        # this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                        # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                        should_unbrake = True

                if should_unbrake:
                    self.unbrake()  # also sets brake_state to NOT_BRAKING

            if self.waiting_for != 0:
                self.last_braking_distance = self.dist_from(self.waiting_for)

    def get_state_dict(self):
        state_dict = {}
        state_dict["center-x"] = self.state.x.x
        state_dict["center-y"] = self.state.x.y
        state_dict["heading"] = self.state.e.psi
        state_dict["corners"] = self.vehicle_body.V
        return state_dict

    def get_other_vehicles(self):
        other_states = {i: self.other_state[i] for i in self.other_state}
        return other_states

    ##### INTENT PREDICTION #####

    def solve_intent_based_control(
        self,
        time=None,
        timer_period=None,
        other_vehicles=[],
        history=None,
        coord_spot_fn=None,
        obstacle_corners={},
        intent_use_obstacles=True,
    ):
        """
        Drive using intent predictions.
        """
        if self.loops <= self.loops_before_predict:
            # initially, just cruise to generate enough history to use intent prediction
            self.solve_task_based_control(time=time, coord_spot_fn=coord_spot_fn)
        else:
            # calculate intent every self.loops_betwen_predict steps (if haven't already decided to park)
            if self.loops % self.loops_between_predict == (
                self.loops_before_predict + 1
            ) % self.loops_between_predict and (
                self.intent is None or self.intent_spot is None
            ):
                self.solve_intent(
                    history, coord_spot_fn, intent_use_obstacles
                )  # populates self.intent

            if self.parking_step == -1:  # if not parking
                done = self.solve_intent_based_control_stanley(
                    time=time,
                    other_vehicles=other_vehicles,
                    coord_spot_fn=coord_spot_fn,
                )

                if done:
                    if self.current_task != "PARK":
                        self.current_task = "PARK"
                        self.set_parking_start_time(time)
                    should_go = self.should_start_parking()

                # determine if need to park
                if (
                    self.intent_spot is not None
                    and self.parking_step == -1
                    and done
                    and should_go
                ):
                    self.parking_step = 0
            else:  # if parking
                # mpc_preds = self.solve_parking_control_mpc(
                #     time=time, obstacle_corners=obstacle_corners
                # )
                # return mpc_preds
                self.solve_parking_control_teleport(time)
                if self.parking_maneuver is None:  # done
                    self.current_task = "END"

        return None

    def solve_intent(self, history, coord_spot_fn, use_obstacles):
        """
        Calculate intent from intent prediction and populate self.intent.
        """
        # get raw intent returned from intent predictor
        predicted_intent = self.intent_sampler.sample_valid_intent(
            self.vehicle_id,
            self.state,
            history,
            coord_spot_fn,
            self.occupancy,
            use_obstacles,
        )

        if (
            predicted_intent is None
        ):  # if predictor did not return a valid intent, keep current intent
            return
        else:
            self.intent = predicted_intent

        # returns None if the intent is not in a spot, else returns the spot number
        in_spot = coord_spot_fn(self.intent)

        # if found a parking spot, set that to the permanent intent
        if in_spot:
            self.intent_spot = in_spot

            # load y-coordinate of nearest waypoint (to tell if parking in a north or south spot)
            nearest_waypoint = self.graph.vertices[
                self.graph.search(self.intent)
            ].coords
            lane_x, lane_y = nearest_waypoint[0], nearest_waypoint[1]

            # determine parameters of parking maneuver
            graph_sol = AStarPlanner(
                self.graph.vertices[
                    self.graph.search([self.state.x.x, self.state.x.y])
                ],
                self.graph.vertices[self.graph.search(self.intent)],
            ).solve()
            if len(graph_sol.vertices) <= 1:
                dir = "east" if self.state.x.x < self.intent[0] else "west"
                xpos = "left" if self.state.x.x < self.intent[0] else "right"
            else:
                approach_left = (
                    graph_sol.vertices[-1].coords[0] > graph_sol.vertices[-2].coords[0]
                )
                dir = "east" if approach_left else "west"
                xpos = "left" if approach_left else "right"
            loc = "north" if lane_y < self.intent[1] else "south"
            heading = "up" if np.random.rand() < 0.5 else "down"
            self.intent_maneuver_params = (dir, xpos, loc, heading)

            # adjust intent for parking
            self.intent[0] += -4 if xpos == "left" else 4
            self.intent[1] = lane_y

            # save path
            x_ref, y_ref, yaw_ref = self.compute_ref_path(graph_sol=graph_sol)
            self.set_ref_pose(x_ref, y_ref, yaw_ref)

            # mark this spot as full
            self.change_central_occupancy(in_spot, True)

    def solve_intent_based_control_stanley(
        self, time=None, other_vehicles=[], coord_spot_fn=None
    ):
        """
        Stanley controller used for intent-based driving. Looks very similar to solve_task_based_control.
        """
        if self.current_task == "END":
            return True

        # Firstly update the set of nearby_vehicles
        self.update_nearby_vehicles()

        # driving control (only need to recalculate path if haven't decided to park yet)
        if self.intent_spot is None:
            graph_sol = AStarPlanner(
                self.graph.vertices[
                    self.graph.search([self.state.x.x, self.state.x.y])
                ],
                self.graph.vertices[self.graph.search(self.intent)],
            ).solve()
            x_ref, y_ref, yaw_ref = self.compute_ref_path(graph_sol=graph_sol)
            self.set_ref_pose(x_ref, y_ref, yaw_ref)

        # check if need to start parking
        if self.reached_target(target=self.intent):
            origin = [
                self.state.x.x + 4
                if self.intent_maneuver_params[1] == "left"
                else self.state.x.x - 4,
                self.state.x.y + self.vehicle_config.parking_start_offset
                if self.intent_maneuver_params[0] == "east"
                else self.state.x.y - self.vehicle_config.parking_start_offset,
                self.state.e.psi,
            ]
            offline_maneuver = self.offline_maneuver.get_maneuver(
                xy_offset=origin,
                driving_dir=self.intent_maneuver_params[0],
                x_position=self.intent_maneuver_params[1],
                spot=self.intent_maneuver_params[2],
                heading=self.intent_maneuver_params[3],
            )
            time_seq = np.arange(
                start=offline_maneuver.t[0],
                stop=offline_maneuver.t[-1],
                step=self.controller.dt,
            )
            self.parking_maneuver = interpolate_states_inputs(
                offline_maneuver, time_seq
            )
            return True

        self.standard_driving_control(coord_spot_fn)

        self.update_state()

        self.state.t = time
        self.state_hist.append(self.state.copy())
        self.logger.append(
            f"t = {time}: x = {self.state.x.x:.2f}, y = {self.state.x.y:.2f}"
        )

        return False

    def solve_parking_control_mpc(
        self,
        time: float,
        obstacle_corners: Dict[Tuple, np.ndarray] = None,
        obstacle_As: List[np.ndarray] = None,
        obstacle_bs: List[np.ndarray] = None,
    ):
        """
        Park by following an offline maneuver using MPC
        """
        if self.parking_start_time == float("inf"):
            self.set_parking_start_time(time)

        parking_time = time - self.parking_start_time
        look_ahead_horizon = np.linspace(
            parking_time,
            parking_time + self.mpc.N * self.mpc.dt,
            num=self.mpc.N,
            endpoint=False,
        )

        ref = VehiclePrediction()

        ref.t = look_ahead_horizon
        ref.x = np.interp(
            ref.t,
            self.parking_maneuver.t,
            self.parking_maneuver.x,
        )
        ref.y = np.interp(
            ref.t,
            self.parking_maneuver.t,
            self.parking_maneuver.y,
        )
        ref.psi = np.interp(
            ref.t,
            self.parking_maneuver.t,
            self.parking_maneuver.psi,
        )
        ref.v = np.interp(
            ref.t,
            self.parking_maneuver.t,
            self.parking_maneuver.v,
        )

        pred = self.mpc.step(
            state=self.state, ref=ref, obstacle_corners=obstacle_corners
        )

        # stop when close enough to final waypoint
        # TODO: better stopping condition
        if (
            np.linalg.norm(
                [
                    self.state.x.x - (self.parking_maneuver.x[-1]),
                    self.state.x.y - (self.parking_maneuver.y[-1]),
                ]
            )
            < 0.3
        ):
            self.current_task = "END"

        return pred
