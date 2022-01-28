from typing import List

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVisualizer

from pathlib import Path

import numpy as np

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody
from parksim.route_planner.a_star import AStarPlanner
from parksim.route_planner.graph import WaypointsGraph
from parksim.path_planner.spline import calc_spline_course
from parksim.path_planner.offline_maneuver import OfflineManeuver
from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle, BrakeState

np.random.seed(6)
# cases and possible solutions
# 44: stopping isn't fast enough for full speed car going toward stopped car
# 5 looks really good

class RuleBasedSimulator(object):
    def __init__(self, dataset: Dataset, offline_maneuver: OfflineManeuver, vis: RealtimeVisualizer):
        self.dlpvis = DlpVisualizer(dataset)
        self.offline_maneuver = offline_maneuver

        self.vis = vis

        self.parking_spaces, self.occupied = self._gen_occupancy()

        self.graph = WaypointsGraph()
        self.graph.setup_with_vis(self.dlpvis)

        # TODO: change these as ROS params
        self.offset = 1.75 # distance off from waypoints
        self.max_target_speed = 3 # for this implementation, normal cruising speed normally was 5

        self.look_ahead_timesteps = 10 # how far to look ahead for crash detection
        self.crash_check_radius = 15 # which vehicles to check crash
        self.parking_radius = 7 # how much room a vehicle should have to park

        # anchor spots
        self.anchor_points = [47, 93, 135, 185, 227, 277, 319, 344] # for now, second spot at the start of a row
        self.anchor_spots = [list(range(21)) + list(range(42, 67)), list(range(21, 42)) + list(range(92, 113)), list(range(67, 92)) + list(range(134, 159)), list(range(113, 134)) + list(range(184, 205)), list(range(159, 184)) + list(range(226, 251)), list(range(205, 226)) + list(range(276, 297)), list(range(251, 276)) + list(range(318, 343)), list(range(297, 318)) + list(range(343, 364))]

        # spawn stuff
        self.spawn_wait = 50 # number of timesteps between cars spawning
        # self.entrance_vertex = 243
        self.entrance_vertex = self.graph.search([14.38, 76.21])

        self.spawn_entering = 3 # number of vehicles to enter
        self.spawn_exiting = 3 # number of vehicles to exit
        self.spawn_exiting_loops = np.random.choice(range(self.spawn_exiting * self.spawn_wait), self.spawn_exiting)

        self.vehicles: List[RuleBasedStanleyVehicle] = []

        self.max_simulation_time = 100

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
    def add_vehicle(self, loops, spot_index):
        north_spot_ranges = [(0, 41), (67, 91), (113, 133), (159, 183), (205, 225), (251, 275), (297, 317)]
        if spot_index > 0: # entering
            north_spot = any([spot_index >= r[0] and spot_index <= r[1] for r in north_spot_ranges])
            y_offset = -5 if north_spot else 5
            coords = [self.parking_spaces[spot_index][0], self.parking_spaces[spot_index][1] + y_offset]
            graph_sol = AStarPlanner(self.graph.vertices[self.entrance_vertex], self.graph.vertices[self.graph.search(coords)]).solve()
        else: # exiting
            north_spot = any([-spot_index >= r[0] and -spot_index <= r[1] for r in north_spot_ranges])
            y_offset = -5 if north_spot else 5
            coords = [self.parking_spaces[-spot_index][0], self.parking_spaces[-spot_index][1] + y_offset]
            print(spot_index, coords)
            graph_sol = AStarPlanner(self.graph.vertices[self.graph.search(coords)], self.graph.vertices[self.entrance_vertex]).solve()

        # collect x, y, yaw from A* solution
        axs = [] 
        ays = []

        # calculate splines
        cxs = []
        cys = []
        cyaws = []

        # generate list of x, y waypoints NOTE: need to cleanup, axs, ays etc. don't need to be global
        axs.append([])
        ays.append([])
        for edge in graph_sol.edges:
            axs[-1].append(edge.v1.coords[0])
            ays[-1].append(edge.v1.coords[1])
        axs[-1].append(graph_sol.edges[-1].v2.coords[0])
        ays[-1].append(graph_sol.edges[-1].v2.coords[1])

        cxs.append([])
        cys.append([])
        cyaws.append([])
        cxs[-1], cys[-1], cyaws[-1], _, _ = calc_spline_course(axs[-1], ays[-1], ds=0.1)
        cxs[-1] = [cxs[-1][j] + self.offset * np.sin(cyaws[-1][j]) for j in range(len(cxs[-1]))]
        cys[-1] = [cys[-1][j] - self.offset * np.cos(cyaws[-1][j]) for j in range(len(cys[-1]))]
        
        initial_state = VehicleState()
        if spot_index > 0: # entering
            initial_state.x.x = cxs[-1][0]
            initial_state.x.y = cys[-1][0]
            initial_state.e.psi = cyaws[-1][0]
        else: # start parked
            # randomize if pointing up or down to start
            initial_state.x.x = self.parking_spaces[-spot_index][0]
            initial_state.x.y = self.parking_spaces[-spot_index][1]
            initial_state.e.psi = np.pi / 2 if np.random.rand() < 0.5 else -np.pi / 2
        
        vehicle = RuleBasedStanleyVehicle(initial_state, VehicleBody(), self.offline_maneuver)
        vehicle.set_ref_pose(cxs[-1], cys[-1], cyaws[-1])
        vehicle.set_ref_v(0)
        vehicle.set_anchor_parking(going_to_anchor=spot_index>0, spot_index=spot_index, should_overshoot=False)

        self.vehicles.append(vehicle)
    
    # starts from an anchor point, goes to an arbitrary spot
    def set_vehicle_park_spot(self, vehicle: RuleBasedStanleyVehicle, new_spot_index):
        
        # go from current location to new spot
        graph_sol = AStarPlanner(self.graph.vertices[self.graph.search([vehicle.state.x.x, vehicle.state.x.y])], self.graph.vertices[self.graph.search(self.parking_spaces[new_spot_index])]).solve()
        new_ax = []
        new_ay = []
        if len(graph_sol.edges) > 0:
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
            pointed_right = vehicle.state.e.psi < np.pi / 2 and vehicle.state.e.psi > -np.pi / 2
        else:
            last_edge = graph_sol.edges[-1]
            pointed_right = last_edge.v2.coords[0] - last_edge.v1.coords[0] > 0

        if pointed_right:
            overshoot_ranges = [(42, 48), (67, 69), (92, 94), (113, 115), (134, 136), (159, 161), (184, 186), (205, 207), (226, 228), (251, 253), (276, 278), (297, 299), (318, 320), (343, 345)]
        else:
            overshoot_ranges = [(64, 66), (89, 91), (156, 158), (181, 183), (248, 250), (273, 275), (340, 342)]

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

        # have the y coordinate of the last waypoint be the same as the new last 
    
        if len(new_ax) == 0:
            new_ay.append(vehicle.y_ref[-1])
        else:
            new_ay.append(new_ay[-1])
        
        # offsets for lanes
        new_cx = []
        new_cy = []
        new_cyaw = []
        new_cx, new_cy, new_cyaw, _, _ = calc_spline_course(new_ax, new_ay, ds=0.1)
        new_cx = [new_cx[j] + self.offset * np.sin(new_cyaw[j]) for j in range(len(new_cx))]
        new_cy = [new_cy[j] - self.offset * np.cos(new_cyaw[j]) for j in range(len(new_cy))]
        
        # set new targets for vehicle
        vehicle.set_ref_pose(new_cx, new_cy, new_cyaw)
        vehicle.set_target_idx(0)
        vehicle.set_anchor_parking(going_to_anchor=False, spot_index=new_spot_index, should_overshoot=should_overshoot)
        
        self.occupied[new_spot_index] = True

    def run(self):
        # while not run out of time and we have not reached the last waypoint yet
        while self.max_simulation_time >= self.time and (len(self.vehicles) < self.spawn_entering + self.spawn_exiting or not all([v.all_done() for v in self.vehicles])):

            # clear visualizer
            self.vis.clear_frame()
            
            # spawn vehicles
            if self.loops % self.spawn_wait == 0 and self.spawn_entering > 0: # entering
                self.add_vehicle(self.loops, np.random.choice(self.anchor_points)) # pick from the anchor points at random
                self.spawn_entering -= 1
            if self.loops in self.spawn_exiting_loops: # spawn in random empty spot
                empty_spots = [i for i in range(len(self.occupied)) if not self.occupied[i]]
                chosen_spot = np.random.choice(empty_spots)
                self.add_vehicle(self.loops, -1 * chosen_spot)
                self.occupied[chosen_spot] = True
                
            for i in range(len(self.vehicles)):
                vehicle = self.vehicles[i]

                # driving control
                
                if vehicle.parking or vehicle.unparking:
                    pass
                elif vehicle.going_to_anchor or not vehicle.reached_target(): # normal driving (haven't reached pre-parking point)
                        
                    # braking controller
                    if vehicle.brake_state == BrakeState.NOT_BRAKING:
                        
                        # normal speed controller if not braking
                        if vehicle.target_idx < vehicle.num_waypoints() - 30:
                            vehicle.v_ref = self.max_target_speed
                        else:
                            vehicle.v_ref = 1

                        # detect parking and unparking
                        # nearby_parkers = [v for v in self.vehicles if (v.mid_park() or v.mid_unpark()) and np.linalg.norm([vehicle.state.x.x - v.state.x.x, vehicle.state.x.y - v.state.x.y]) < self.parking_radius]
                        nearby_parkers = [v for v in self.vehicles if (v.mid_park() or v.mid_unpark()) and vehicle.other_within_parking_box(v) and v is not vehicle]
                        if len(nearby_parkers) > 0: # should only be one nearby parker, since they wait for each other
                            parker = nearby_parkers[0]
                            vehicle.brake(brake_state=BrakeState.WAITING)
                            vehicle.waiting_for = parker
                            vehicle.priority = -1
                            if parker.unparking:
                                vehicle.waiting_for_unparker = True
                            
                        else: # not parking
                            
                            # don't check for crash with self, or vehicles that are all done
                            others = [veh for veh in (self.vehicles[:i] + self.vehicles[i+1:]) if not veh.all_done()]
                            
                            vehicle.crash_set.update(vehicle.will_crash(others, self.look_ahead_timesteps, radius=self.crash_check_radius))
                            # if will crash
                            if len(vehicle.crash_set) > 0:
                                # add ourselves to the crash set
                                vehicle.crash_set.add(vehicle)
                                # add ourselves to other vehicle crash sets (to cause them to stop)
                                for c in vehicle.crash_set:
                                    c.crash_set.add(vehicle)

                                # recursively add all that they will also crash with to our set
                                secondary_crash_set = set()
                                old_len = 0
                                new_len = 1 # just to make sure the loop runs at least once

                                # keep checking until no longer adding vehicles
                                while new_len - old_len > 0:
                                    old_len = len(secondary_crash_set)
                                    for v in vehicle.crash_set:
                                        secondary_crash_set.update(v.crash_set)
                                    new_len = len(secondary_crash_set)
                                    vehicle.crash_set.update(secondary_crash_set)

                                # variable to tell where to go next (default is braking)
                                next_state = BrakeState.BRAKING

                                # set priority if not already set for this vehicle
                                if vehicle.priority is None:
                                    # for leading/trailing situation
                                    lst = list(vehicle.crash_set)
                                    # ang = ((lst[0].state.e.psi - lst[1].state.e.psi) + (2 * np.pi)) % (2 * np.pi) # [0, 2pi)
                                    # if len(vehicle.crash_set) == 2 and (ang < 0.25 or ang > 2 * np.pi - 0.25): # about 15 degrees, should make this a constant
                                    if len(vehicle.crash_set) == 2: # this means that if there are only 2 cars crashing, one won't stop. That's unrealistic, probably would need some constraint on angle
                                        this_v = lst[0] if lst[0] is vehicle else lst[1]
                                        other_v = lst[1] if lst[0] is vehicle else lst[0]
                                        # this_to_other_ang = ((np.arctan2(other_v.state.x.y - this_v.state.x.y, other_v.state.x.x - this_v.state.x.x) - other_v.state.e.psi) + (2*np.pi)) % (2*np.pi)
                                        # if this_to_other_ang < np.pi / 2 or this_to_other_ang > 3 * np.pi / 2: # trailing car
                                        if not this_v.should_go_before(other_v):
                                            next_state = BrakeState.WAITING # go straight to waiting, no priority calculations necessary
                                            vehicle.priority = other_v.priority - 1 if other_v.priority is not None else -1 # so cars that may brake behind it can have a priority
                                            vehicle.waiting_for = other_v
                                        else: # leading car
                                            next_state = BrakeState.NOT_BRAKING # don't brake
                                            vehicle.crash_set.clear() # not going to crash anymore   
                                    else:
                                        # if this is first detection of collision
                                        # NOTE: any priorities set in here should be between 0 (inclusive) and 1 (exclusive)
                                        if all([not v.braking() for v in vehicle.crash_set if v is not vehicle]):
                                            for c in vehicle.crash_set:
                                                c.priority = np.random.rand()
                                        else: # new car meeting up with cars that have already braked
                                            # wait for last car in queue
                                            next_state = BrakeState.WAITING
                                            vehicle.waiting_for = min([v for v in vehicle.crash_set if v is not vehicle], key=lambda v: v.priority)
                                            vehicle.priority = min([v.priority for v in vehicle.crash_set if v is not vehicle]) - 1

                                if next_state != BrakeState.NOT_BRAKING:
                                    vehicle.brake(brake_state=next_state) 

                    elif vehicle.brake_state == BrakeState.BRAKING: # when we don't know who we're waiting for yet, but know we need to brake
                
                        # don't check for crash with self, or vehicles that are all done
                        others = [veh for veh in (self.vehicles[:i] + self.vehicles[i+1:]) if not veh.all_done()]
                        
                        # add more vehicles that will crash with if necessary
                        crashers = vehicle.will_crash(others, self.look_ahead_timesteps, radius=self.crash_check_radius)
                        if len(crashers) > 0: # if for efficiency
                            vehicle.crash_set.update(crashers)
                            # add ourselves to other vehicle crash sets (to cause them to stop)
                            for c in crashers:
                                c.crash_set.add(vehicle)
                            
                        
                        # recursively add all that they will also crash with to our set
                        secondary_crash_set = set()
                        old_len = 0
                        new_len = 1 # just to make sure the loop runs at least once

                        # keep checking until no longer adding vehicles
                        while new_len - old_len > 0:
                            old_len = len(secondary_crash_set)
                            for v in vehicle.crash_set:
                                secondary_crash_set.update(v.crash_set)
                            new_len = len(secondary_crash_set)
                            vehicle.crash_set.update(secondary_crash_set)

                        # if everyone stopped
                        if all([crasher.state.v.v < 0.05 for crasher in vehicle.crash_set]):
                            # determine order of going (for now random)
                            order = sorted(list(vehicle.crash_set), key=lambda o: o.priority, reverse=True)
                            # determine who we're waiting for
                            if order[0] == vehicle: # we go first
                                vehicle.waiting_for = None
                            else: # we're waiting
                                for oi in range(1, len(order)):
                                    if order[oi] == vehicle:
                                        vehicle.waiting_for = order[oi - 1]
                            
                            vehicle.brake_state = BrakeState.WAITING

                    else: # waiting
                
                        # parking
                        if vehicle.waiting_for is not None and vehicle.waiting_for.parking:
                            if vehicle.waiting_for.all_done():
                                vehicle.unbrake()
                                
                        elif vehicle.waiting_for is not None and vehicle.waiting_for_unparker:
                            if not vehicle.waiting_for.unparking:
                                vehicle.waiting_for_unparker = False
                                vehicle.unbrake()
                                
                        else:

                            # other (standard) cases

                            should_unbrake = False

                            # go if going first
                            if vehicle.waiting_for is None:
                                should_unbrake = True
                            else:
                                # TODO: better heuristic for unbraking
                                if vehicle.waiting_for.all_done() or (not vehicle.waiting_for.braking() and vehicle.waiting_for.has_passed(vehicle)) or np.linalg.norm([vehicle.waiting_for.state.x.x - vehicle.state.x.x, vehicle.waiting_for.state.x.y - vehicle.state.x.y]) > 10:
                                    should_unbrake = True
                                elif vehicle.waiting_for.waiting_for == vehicle: # if the vehicle you're waiting for is waiting for you
                                    # you should go
                                    # NOTE: this line could be an issue if the vehicles aren't checked in a loop (since either could go)
                                    # But for now, since it is checked in a loop, once a vehicle is set to waiting, the other vehicle is guaranteed to be checked before this vehicle is checked again
                                    should_unbrake = True

                            if should_unbrake:
                                # remove vehicle from other vehicle's crash sets
                                for c in vehicle.crash_set:
                                    if c != vehicle and vehicle in c.crash_set:
                                        c.crash_set.remove(vehicle)

                                vehicle.unbrake() # also sets brake_state to NOT_BRAKING
                
                    # if gotten near anchor point, figure out where to go next
                    if vehicle.going_to_anchor and vehicle.spot_index > 0 and vehicle.target_idx > vehicle.num_waypoints() - 10:
                        possible_spots = [i for i in self.anchor_spots[self.anchor_points.index(vehicle.spot_index)] if not self.occupied[i]]
                        self.set_vehicle_park_spot(vehicle, np.random.choice(possible_spots))

                else:
                    # if reached target (pre-parking point), start parking
                    vehicle.v_ref = 0
                    vehicle.parking = vehicle.spot_index > 0 # park unless going to exit
                    # NOTE: if vehicle.spot_index < 0 (i.e. exiting), then kill the node b/c we're done
                    
                if vehicle.parking: # wait for coast to be clear, then start parking
                    # everyone within range should be braking or parking or unparking
                    # NOTE: this doesn't yet account for a braked vehicle in the way of our parking
                    should_go = all([v.all_done() or (v.parking and v.parking_start_time > vehicle.parking_start_time) or (v.unparking and v.unparking_start_time > vehicle.unparking_start_time) or v.braking() or np.linalg.norm([vehicle.state.x.x - v.state.x.x, vehicle.state.x.y - v.state.x.y]) >= self.parking_radius * 2 for v in self.vehicles if v != vehicle])
                    if vehicle.park_start_coords is None:
                        vehicle.park_start_coords = (vehicle.state.x.x - self.offset * np.sin(vehicle.state.e.psi), vehicle.state.x.y + self.offset * np.cos(vehicle.state.e.psi))
                    vehicle.update_state_parking(self.time, should_go)
                elif vehicle.unparking: # wait for coast to be clear, then start unparking
                    # everyone within range should be braking or parking or unparking
                    should_go = all([v.all_done() or (v.parking and v.parking_start_time > vehicle.parking_start_time) or (v.unparking and v.unparking_start_time > vehicle.unparking_start_time) or v.braking() or np.linalg.norm([vehicle.state.x.x - v.state.x.x, vehicle.state.x.y - v.state.x.y]) >= self.parking_radius * 2 for v in self.vehicles if v != vehicle])
                    vehicle.update_state_unparking(self.time, should_go)
                else: 
                    vehicle.update_state(self.time)

            self.time += 0.1
            self.loops += 1

            # Visualize
            for vehicle in self.vehicles:
                if vehicle.all_done():
                    fill = (0, 0, 0, 255)
                elif vehicle.braking():
                    fill = (255, 0, 0, 255)
                elif vehicle.parking or vehicle.unparking:
                    fill = (255, 128, 0, 255)
                else:
                    fill = (0, 255, 0, 255)

                self.vis.draw_vehicle(states=[vehicle.state.x.x, vehicle.state.x.y, vehicle.state.e.psi], fill=fill)
                self.vis.draw_line(points=np.array([vehicle.x_ref, vehicle.y_ref]).T, color=(39,228,245, 193))
                on_vehicle_text = vehicle.unparking_step
                self.vis.draw_text([vehicle.state.x.x - 2, vehicle.state.x.y + 2], on_vehicle_text, size=25)
            self.vis.render()

def main():
    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    print('Loading dataset...')
    # ds.load(home_path + '/dlp-dataset/data/DJI_0012')
    ds.load(home_path + '/Documents/Berkeley/Research/dlp-dataset/data/DJI_0012')
    print("Dataset loaded.")

    # offline_maneuver = OfflineManeuver(pickle_file=home_path + '/ParkSim/parking_maneuvers.pickle')
    offline_maneuver = OfflineManeuver(pickle_file=home_path + '/Documents/Berkeley/Research/ParkSim/parking_maneuvers.pickle')

    vis = RealtimeVisualizer(ds, VehicleBody())

    simulator = RuleBasedSimulator(ds, offline_maneuver, vis)

    simulator.run()



if __name__ == "__main__":
    main()