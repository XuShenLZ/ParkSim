from typing import List, Set
import numpy as np
from pytope import Polytope
from enum import Enum

from parksim.agents.abstract_agent import AbstractAgent
from parksim.controller.stanley_controller import StanleyController

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

import os
import contextlib

class BrakeState(Enum):
    NOT_BRAKING = 0
    BRAKING = 1
    WAITING = 2

class RuleBasedStanleyVehicle(AbstractAgent):
    def __init__(self, x_waypoints, y_waypoints, yaw_waypoints, initial_state: VehicleState, vehicle_body: VehicleBody, spot_index, should_overshoot, offline_maneuver, target_speed=0, visual_metadata=0, controller: StanleyController = StanleyController()):

        self.x_waypoints = x_waypoints # x coordinates for waypoints
        self.y_waypoints = y_waypoints # y coordinates for waypoints
        self.yaw_waypoints = yaw_waypoints # yaws for waypoints
        self.state = initial_state # state

        self.controller = controller

        self.target_idx = self.controller.calc_target_index(self.state, self.x_waypoints, self.y_waypoints)[0] # waypoint the vehicle is targeting
        self.last_idx = len(self.x_waypoints) - 1 # terminal waypoint
        self.visited_indices = [] # waypoints the vehicle has targeted
        self.visited_speed = [] # speed the vehical has gone

        self.vehicle_body = vehicle_body # dimensions

        self.target_speed = target_speed # target speed
        
        self.reached_tgt = False
        
        # parking stuff
        self.going_to_anchor = spot_index > 0 # going to anchor if parking, not if exiting
        self.spot_index = spot_index
        self.should_overshoot = should_overshoot # overshooting or undershooting the spot?
        self.park_start_coords = None
        self.offline_maneuver = offline_maneuver
        self.parking = False # this is True after we have started parking, including when we're done
        self.parking_maneuver_state = None
        self.parking_maneuver_index = None
        self.parking_step = 0
        
        # unparking stuff
        self.unparking = self.spot_index < 0 # are we waiting to unpark or are currently unparking?
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
        
        # visualization
        self.visual_metadata = visual_metadata # right now, just has loop that this car starts on, could contain more stuff
        self.visited_x = [initial_state.x.x] # x coordinates the vehicle has travelled
        self.visited_y = [initial_state.x.y] # y coordinates the vehicle has travelled
        self.visited_yaw = [initial_state.e.psi] # yaw for where the vehicle has travelled
        self.visited_v = [initial_state.v.v] # velocity for where the vehicle has travelled
        self.visited_t = [0.0] # times for where the vehicle has travelled
        self.visited_braking = [False] # were we braking at this time?
        self.visited_parking = [self.parking or self.unparking] # were we parking or unparking at this time?

    def reached_target(self):
        # return self.last_idx == self.target_idx
        # need to constantize this
        if not self.reached_tgt:
            self.reached_tgt = np.linalg.norm([self.state.x.x - self.x_waypoints[-1], self.state.x.y - self.y_waypoints[-1]]) < 0.3
        return self.reached_tgt

    def num_waypoints(self):
        return len(self.x_waypoints)

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

            ai = self.controller.pid_control(self.target_speed, look_ahead_state.v.v, braking=self.braking())
            di, _ = self.controller.stanley_control(look_ahead_state, self.x_waypoints, self.y_waypoints, self.yaw_waypoints, self.target_idx)
            self.controller.step(look_ahead_state, ai, di)

            for v in range(len(other_vehicles)):
                if other_vehicles[v] not in will_crash_with: # for efficiency
                    ai = self.controller.pid_control(other_vehicles[v].target_speed, other_look_ahead_states[v].v.v, braking=other_vehicles[v].braking()) 
                    di, _ = self.controller.stanley_control(other_look_ahead_states[v], other_vehicles[v].x_waypoints, other_vehicles[v].y_waypoints, other_vehicles[v].yaw_waypoints, other_vehicles[v].target_idx)
                    self.controller.step(other_look_ahead_states[v], ai, di)

            # detect crash

            this_polytope = Polytope(self.get_corners(look_ahead_state))

            for v in range(len(other_vehicles)):
                if other_vehicles[v] not in will_crash_with: # for efficiency
                    other_polytope = Polytope(self.get_corners(other_look_ahead_states[v]))

                    # inter_polytope = None
                    # with io.capture_output() as captured:
                    #     inter_polytope = this_polytope & other_polytope
                    inter_polytope = None
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        inter_polytope = this_polytope & other_polytope
                    if len(inter_polytope.V) > 0: # they crash
                        will_crash_with.add(other_vehicles[v])
        return will_crash_with

    def update_state(self, time):
        # get acceleration toward target speed (ai)
        ai = self.controller.pid_control(self.target_speed, self.state.v.v, braking=self.braking()) 
        # get amount we should turn (di) and next target (target_idx)
        di, self.target_idx = self.controller.stanley_control(self.state, self.x_waypoints, self.y_waypoints, self.yaw_waypoints, self.target_idx)
        # advance state of vehicle (updates x, y, yaw, velocity)
        self.controller.step(self.state, ai, di)

        # add to list of states
        self.visited_x.append(self.state.x.x)
        self.visited_y.append(self.state.x.y)
        self.visited_yaw.append(self.state.e.psi)
        self.visited_v.append(self.state.v.v)
        self.visited_t.append(time)
        self.visited_braking.append(self.braking())
        self.visited_parking.append(False)

        if len(self.visited_indices) == 0 or self.visited_indices[-1] != self.target_idx:
            self.visited_indices.append(self.target_idx)
            self.visited_speed.append(self.state.v.v)
            
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
        
        # add to state history
        self.visited_x.append(self.state.x.x)
        self.visited_y.append(self.state.x.y)
        self.visited_yaw.append(self.state.e.psi)
        self.visited_v.append(self.state.v.v)
        self.visited_t.append(time)
        self.visited_braking.append(False)
        self.visited_parking.append(True)
        
        # update parking step if advancing
        self.parking_step += 1 if advance else 0
        
    def update_state_unparking(self, time, advance=True):
        if self.unparking_maneuver_state is None: # start unparking
            # get unparking parameters
            direction = 'west' if self.x_waypoints[0] > self.x_waypoints[1] else 'east' # if first direction of travel is left, face west
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
        
        # add to state history
        self.visited_x.append(self.state.x.x)
        self.visited_y.append(self.state.x.y)
        self.visited_yaw.append(self.state.e.psi)
        self.visited_v.append(self.state.v.v)
        self.visited_t.append(time)
        self.visited_braking.append(False)
        self.visited_parking.append(True)
        
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
        self._pre_brake_target_speed = self.target_speed
        self.target_speed = 0
        self.brake_state = brake_state

    def unbrake(self):
        """
        Set target speed back to what it was. Only does something if braking
        """
        if self.brake_state != BrakeState.NOT_BRAKING:
            self.target_speed = self._pre_brake_target_speed
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