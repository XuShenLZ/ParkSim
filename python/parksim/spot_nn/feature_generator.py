from typing import List
import torch
import numpy as np

from parksim.pytypes import VehicleState
from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

class SpotFeatureGenerator():

    @staticmethod
    def generate_features(vehicle: RuleBasedStanleyVehicle, active_vehicles: List[RuleBasedStanleyVehicle], spawn_mean: int):
        heatmap = SpotFeatureGenerator.generate_vehicle_heatmap(active_vehicles)
        squares_traveled = set()
        for i in range(len(vehicle.controller.x_ref)):
            squares_traveled.add(SpotFeatureGenerator.coord_to_heatmap_square(vehicle.controller.x_ref[i], vehicle.controller.y_ref[i]))
        # subtract one because you're in your own path always
        vehicles_along_path = sum([heatmap[sq] for sq in squares_traveled]) - 1
        # subtract one because you're parking near yourself
        vehicles_parking_nearby = sum([np.linalg.norm([v.controller.x_ref[-1] - vehicle.controller.x_ref[-1], v.controller.y_ref[-1] - vehicle.controller.y_ref[-1]]) < 10 for v in active_vehicles]) - 1

        return torch.FloatTensor([vehicle.controller.get_ref_length(), vehicle.parking_spaces[vehicle.spot_index][0], vehicle.parking_spaces[vehicle.spot_index][1], vehicles_along_path, vehicles_parking_nearby, spawn_mean])

    @staticmethod
    def number_of_features():
        return 6

    @staticmethod
    def generate_vehicle_heatmap(vehicles: List[RuleBasedStanleyVehicle]) -> List[int]:
        heatmap = [0] * 112
        for v in vehicles:
            if v.state.x.x < 0 or v.state.x.x >= 140 or v.state.x.y < 0 or v.state.x.y >= 80:
                continue
            heatmap[SpotFeatureGenerator.coord_to_heatmap_square(v.state.x.x, v.state.x.y)] += 1
        return heatmap

    @staticmethod
    def coord_to_heatmap_square(x, y):
        return int(x // 10) * 8 + int(y // 10)
