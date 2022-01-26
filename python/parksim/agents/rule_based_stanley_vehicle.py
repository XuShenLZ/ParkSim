from typing import List, Set
import numpy as np
from parksim.path_planner.offline_maneuver import OfflineManeuver
from enum import Enum

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

class BrakeState(Enum):
    NOT_BRAKING = 0
    BRAKING = 1
    WAITING = 2

class RuleBasedStanleyVehicle(AbstractAgent):
    def __init__(self, initial_state: VehicleState, vehicle_body: VehicleBody, offline_maneuver: OfflineManeuver, controller: StanleyController = StanleyController(), predictor: StanleyController = StanleyController()):

        # State and Reference Waypoints
        self.state = initial_state # state

        self.x_ref = [] # x coordinates for waypoints
        self.y_ref = [] # y coordinates for waypoints
        self.yaw_ref = [] # yaws for waypoints
        self.v_ref = 0 # target speed

        # Dimensions
        self.vehicle_body = vehicle_body

        # Controller and Predictor
        self.controller = controller
        self.predictor = predictor

        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        # self.target_idx = self.controller.calc_target_index(self.state)[0] # waypoint the vehicle is targeting
        self.target_idx = 0

        self.last_idx = len(self.x_ref) - 1 # terminal waypoint
        self.visited_indices = [] # waypoints the vehicle has targeted
        self.visited_speed = [] # speed the vehical has gone
        
        self.reached_tgt = False
        
        # parking stuff
        self.going_to_anchor = True # going to anchor if parking, not if exiting
        self.spot_index = 0
        self.should_overshoot = False # overshooting or undershooting the spot?
        self.park_start_coords = None
        self.offline_maneuver = offline_maneuver
        self.parking = False # this is True after we have started parking, including when we're done
        self.parking_maneuver_state = None
        self.parking_maneuver_index = None
        self.parking_step = 0
        
        # unparking stuff
        self.unparking = False
        self.unparking_maneuver_state = None
        self.unparking_maneuver_index = None
        self.unparking_step = 0
        
        # braking stuff
        self.brake_state = BrakeState.NOT_BRAKING # are we braking?
        self._pre_brake_target_speed = 0 # speed to restore when unbraking
        self.priority = None # priority for going after braking
        self.crash_set: Set[RuleBasedStanleyVehicle] = set() # vehicles that we will crash with
        self.waiting_for: RuleBasedStanleyVehicle = None # vehicle waiting for before we go
        self.waiting_for_unparker = False # need special handling for waiting for unparker

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

    def set_anchor_parking(self, going_to_anchor, spot_index, should_overshoot):
        self.going_to_anchor = going_to_anchor
        self.spot_index = spot_index
        self.should_overshoot = should_overshoot
        self.unparking = self.spot_index < 0 # are we waiting to unpark or are currently unparking?

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

    def will_crash(self, other_vehicles: List[AbstractAgent], look_ahead_timesteps, radius=None, verbose=False):
        
        will_crash_with = set()
        
        if radius is not None:
            other_vehicles = [v for v in other_vehicles if np.linalg.norm([self.state.x.x - v.state.x.x, self.state.x.y - v.state.x.y]) < radius]
        
        # create states for looking ahead
        look_ahead_state = self.state.copy()
        other_look_ahead_states = [v.state.copy() for v in other_vehicles]

        # for each time step, looking ahead
        for i in range(look_ahead_timesteps):

            # calculate new positions
            self.predictor.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
            self.predictor.set_ref_v(self.v_ref)
            self.predictor.set_target_idx(self.target_idx)
            ai, di, _ = self.predictor.solve(look_ahead_state, self.braking())
            self.predictor.step(look_ahead_state, ai, di)

            for v in range(len(other_vehicles)):
                if other_vehicles[v] not in will_crash_with: # for efficiency
                    other_vehicle = other_vehicles[v]
                    other_look_ahead_state = other_look_ahead_states[v]
                    self.predictor.set_ref_pose(other_vehicle.x_ref, other_vehicle.y_ref, other_vehicle.yaw_ref)
                    self.predictor.set_ref_v(other_vehicle.v_ref)
                    self.predictor.set_target_idx(other_vehicle.target_idx)
                    ai, di, _ = self.predictor.solve(other_look_ahead_state, other_vehicle.braking())
                    self.predictor.step(other_look_ahead_state, ai, di)


            # detect crash
            for v in range(len(other_vehicles)):
                if other_vehicles[v] not in will_crash_with: # for efficiency
                    if self.will_collide(other_look_ahead_states[v], other_vehicles[v].vehicle_body):
                        will_crash_with.add(other_vehicles[v])


        return will_crash_with

    def update_state(self, time):
        self.controller.set_ref_pose(self.x_ref, self.y_ref, self.yaw_ref)
        self.controller.set_ref_v(self.v_ref)
        self.controller.set_target_idx(self.target_idx)
        # get acceleration toward target speed (ai), amount we should turn (di), and next target (target_idx)
        ai, di, self.target_idx = self.controller.solve(self.state, self.braking())
        # advance state of vehicle (updates x, y, yaw, velocity)
        self.controller.step(self.state, ai, di)
            
    def update_state_parking(self, time, advance=True):
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
        
    def update_state_unparking(self, time, advance=True):
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
            self.unparking = False
        else:
            # update parking step if advancing
            self.unparking_step -= 1 if advance else 0
        
    def get_corners(self, state: VehicleState=None, vehicle_body: VehicleBody=None):
        """
        center: center of vehicle (x, y)
        dims: dimensions of vehicle (length, width)
        angle: angle of vehicle in radians
        """
        if state is None:
            state = self.state

        if vehicle_body is None:
            vehicle_body = self.vehicle_body

        center = [state.x.x, state.x.y]
        angle = state.e.psi

        length, width = vehicle_body.l, vehicle_body.w
        # TODO: change these to vehicle_body.V
        offsets = np.array([[ 0.5, -0.5],
                            [ 0.5,  0.5],
                            [-0.5,  0.5],
                            [-0.5, -0.5]])
        offsets_scaled = offsets @ np.array([[length, 0], [0, width]])

        adj_angle = np.pi - angle
        c, s = np.cos(adj_angle), np.sin(adj_angle)
        rot_mat = np.array([[c, s], [-s, c]])
        offsets_rotated = rot_mat @ offsets_scaled.T

        c = np.array([*center])
        c_stacked = np.vstack((c, c, c, c))
        return offsets_rotated.T + c_stacked

    def brake(self, brake_state=BrakeState.BRAKING):
        """
        Set target speed to 0 and turn on brakes, which make deceleration faster
        """
        self._pre_brake_target_speed = self.v_ref
        self.v_ref = 0
        self.brake_state = brake_state

    def unbrake(self):
        """
        Set target speed back to what it was. Only does something if braking
        """
        if self.brake_state != BrakeState.NOT_BRAKING:
            self.v_ref = self._pre_brake_target_speed
            self.brake_state = BrakeState.NOT_BRAKING
            self.crash_set.clear()
            self.priority = None
            self.waiting_for = None

    def braking(self):
        """
        Are we braking?
        """
        return self.brake_state != BrakeState.NOT_BRAKING
    
    def currently_unparking(self):
        """
        Have we started the unparking maneuver yet? If this is False, traffic should have the right of way, else this vehicle should have the right of way
        """
        return self.unparking and self.unparking_maneuver_state is not None and self.unparking_step < len(self.unparking_maneuver_state['x']) - 1
    
    def all_done(self):
        """
        Have we finished the parking maneuver or we have reached the exit?
        """
        
        return (self.parking and self.parking_maneuver_state is not None and self.parking_step >= len(self.parking_maneuver_state['x'])) or (self.spot_index < 0 and self.reached_tgt)