from abc import abstractmethod, ABC
from typing import List

import numpy as np

from parksim.utils.rectangle_to_circles import v2c

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

from itertools import product

class AbstractAgent(ABC):
    """
    Abstract Agent Class
    """
    def __init__(self, vehicle_id: int, state: VehicleState, vehicle_body: VehicleBody):
        self.vehicle_id = vehicle_id
        
        self.state = state
        self.vehicle_body = vehicle_body

        self.x_ref: List[float] = []
        self.y_ref: List[float] = []
        self.yaw_ref: List[float] = []
        self.v_ref: float = 0.0

    def _v2c(self, state: VehicleState, vehicle_body: VehicleBody):
        """
        Use a few circles to approximate vehicle body rectangle
        num_circles: the number of circles to approximate
        """
        return v2c(state=state, vehicle_body=vehicle_body)

    def will_collide(self, state: VehicleState, vehicle_body: VehicleBody) -> bool:
        """
        Check collision using circles. Return True if will collide
        
        state: The state of the other vehicle
        vehicle_body: The vehicle body of the other vehicle
        eps: A tunable safety margin
        """
        circles_self = self._v2c(self.state, self.vehicle_body)
        circles_other = self._v2c(state, vehicle_body)

        for circle_a, circle_b in product(circles_self, circles_other):
            dist = np.linalg.norm([circle_a[0]-circle_b[0], circle_a[1]-circle_b[1]])
            if dist < circle_a[2] + circle_b[2]:
                return True

        return False
