import pickle
from typing import List
import torch
import numpy as np
from pathlib import Path

from parksim.pytypes import VehicleState
from parksim.agents.rule_based_vehicle import RuleBasedVehicle


class SpotFeatureGenerator:
    def __init__(self):
        with open(
            str(Path.home())
            + "/ParkSim/python/parksim/spot_nn/create_features_data.pickle",
            "rb",
        ) as file:
            self.create_features_data = pickle.load(file)
        with open(str(Path.home()) + "/ParkSim/data/spots_data.pickle", "rb") as f:
            data = pickle.load(f)
            self.parking_spaces = data["parking_spaces"]

    def generate_features(
        self,
        spot_index: int,
        active_vehicles: List[RuleBasedVehicle],
        spawn_mean: int,
        queue: int,
    ):
        heatmap = self.generate_vehicle_heatmap(active_vehicles)
        # subtract one because you're in your own path always
        vehicles_along_path = (
            sum(
                [
                    heatmap[sq]
                    for sq in self.create_features_data["trajectory_squares"][
                        spot_index
                    ]
                ]
            )
            - 1
        )
        # subtract one because you're parking near yourself
        vehicles_parking_nearby = (
            sum(
                [
                    np.linalg.norm(
                        [
                            v.controller.x_ref[-1]
                            - self.create_features_data["last_waypoint"][spot_index][0],
                            v.controller.y_ref[-1]
                            - self.create_features_data["last_waypoint"][spot_index][1],
                        ]
                    )
                    < 10
                    for v in active_vehicles
                    if len(v.controller.x_ref) > 0
                ]
            )
            - 1
        )

        feat = torch.FloatTensor(
            [
                self.create_features_data["trajectory_length"][spot_index],
                self.parking_spaces[spot_index][0],
                self.parking_spaces[spot_index][1],
                vehicles_along_path,
                vehicles_parking_nearby,
                spawn_mean,
                queue,
            ]
        )
        assert len(feat) == self.number_of_features
        return feat

    @property
    def number_of_features(self):
        return 7

    def generate_vehicle_heatmap(self, vehicles: List[RuleBasedVehicle]) -> List[int]:
        heatmap = [0] * self.number_of_heatmap_squares()
        for v in vehicles:
            if (
                v.state.x.x < 0
                or v.state.x.x >= 140
                or v.state.x.y < 0
                or v.state.x.y >= 80
            ):
                continue
            heatmap[self.coord_to_heatmap_square(v.state.x.x, v.state.x.y)] += 1
        return heatmap

    def number_of_heatmap_squares(self):
        return 221

    def coord_to_heatmap_square(self, x, y):
        if x < 7.71:
            horz = 0
        elif x < 76.54:
            horz = 1 + (x - 7.71) // 8.60375  # divide up into 8 boxes
        elif x < 83.83:
            horz = 9
        elif x < 138.42:
            horz = 10 + (x - 83.82) // 9.1  # divide up into 6 boxes
        else:
            horz = 16

        if y < 6.48:
            vert = 0
        elif y < 13.51:
            vert = 1
        elif y < 19.095:
            vert = 2
        elif y < 24.68:
            vert = 3
        elif y < 31.93:
            vert = 4
        elif y < 37.585:
            vert = 5
        elif y < 43.24:
            vert = 6
        elif y < 50.4:
            vert = 7
        elif y < 55.9:
            vert = 8
        elif y < 61.4:
            vert = 9
        elif y < 68.51:
            vert = 10
        elif y < 73.73:
            vert = 11
        else:
            vert = 12

        return int(horz * 13 + vert)
