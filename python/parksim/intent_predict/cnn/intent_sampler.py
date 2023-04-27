from typing import List, Tuple
import numpy as np
import PIL
import heapq

import torch
from torchvision import transforms

from dlp.visualizer import SemanticVisualizer

from parksim.pytypes import VehicleState

from parksim.route_planner.graph import Vertex, WaypointsGraph

from parksim.intent_predict.cnn.predictor import Predictor, PredictionResponse
from parksim.intent_predict.cnn.visualizer.instance_centric_generator import InstanceCentricGenerator
from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from parksim.spot_detector.detector import LocalDetector

class IntentSampler():

    def __init__(self, inst_centric_generator=InstanceCentricGenerator(), intent_predictor=Predictor(), intent_extractor: CNNDataProcessor=None, spot_detector: LocalDetector = None):
        self.inst_centric_generator = inst_centric_generator
        self.intent_predictor = intent_predictor
        self.intent_extractor = intent_extractor
        self.spot_detector = spot_detector

        self.intent_graph = WaypointsGraph()
        self.intent_graph.setup_with_vis(self.intent_extractor.vis)

    def sample_valid_intent(self, vehicle_id: int, state: VehicleState, history, coord_spot_fn, occupancy, use_obstacles):
        """
        Find most likely feasible intents from intent predictor, then pick one.
        """
        # run intent predictor and load most popular intents

        intents = self.run_intent_predictor(vehicle_id, state, history, occupancy, use_obstacles)
        best_lanes = self.find_n_best_lanes(
            [state.x.x, state.x.y],
            state.e.psi,
            graph=self.intent_graph,
            vis=self.intent_extractor.vis,
            predictor=self.intent_predictor,
        )

        distributions, coordinates = self.expand_distribution(intents, best_lanes)

        top_n = list(zip(distributions, coordinates))
        top_n.sort(reverse=True)

        # find valid intents

        valid_probs = []
        valid_coords = []

        for prob, coords in top_n:
            in_spot = coord_spot_fn(coords)  # determine if intent is in a spot

            # can't have an intent behind you
            ang = (
                (
                    np.arctan2(coords[1] - state.x.y, coords[0] - state.x.x)
                    - state.e.psi
                )
                + (2 * np.pi)
            ) % (
                2 * np.pi
            )  # between 0 and 2pi
            if ang > np.pi / 2 - np.pi / 6 and ang < 3 * np.pi / 2 + np.pi / 6:
                continue

            # intent can't be near you (to prevent stagnation)
            if (
                np.linalg.norm([state.x.x - coords[0], state.x.y - coords[1]])
                < 4
            ):
                continue

            # can't go to already occupied space
            if in_spot and occupancy[in_spot]:
                continue

            if state.x.x > 10 and state.x.x < 50 and state.x.y > 60 and state.x.y < 70 and coords[0] > 10 and coords[0] < 50 and coords[1] > 40 and coords[1] < 55:
                continue

            valid_probs.append(prob)
            valid_coords.append(coords)

        # if no valid intents, return None (which leaves self.intent the same)
        if len(valid_coords) == 0:
            return None

        return valid_coords[
            np.random.choice(
                range(len(valid_coords)), p=np.array(valid_probs) / sum(valid_probs)
            )
        ]

    def run_intent_predictor(self, vehicle_id: int, state: VehicleState, history, occupancy: List[bool], use_obstacles: bool):
        """
        Predict the most likely intents of the specific vehicle.
        """
        img = self.inst_centric_generator.inst_centric(vehicle_id, history, occupancy, use_obstacles)
        return self.intent_predictor.predict(
            img,
            np.array([state.x.x, state.x.y]),
            state.e.psi,
            state.v.v,
            1.0,
        )

    def find_n_best_lanes(
        self,
        start_coords,
        global_heading,
        graph: WaypointsGraph,
        vis: SemanticVisualizer,
        predictor: Predictor,
        n=3,
    ):
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
                    if vis._is_visible(
                        current_state=current_state, target_state=child.coords
                    ):
                        all_lanes.add(child)
                continue

            for child in children:
                if child not in visited:
                    fringe.append(child)

        lanes = []
        for i, lane in enumerate(all_lanes):
            astar_dist, astar_dir = predictor.compute_Astar_dist_dir(
                current_state, lane.coords, global_heading
            )
            heapq.heappush(
                lanes, (-astar_dir, astar_dist, i, lane.coords)
            )  # i is to avoid issues when two heap elements are the same

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
