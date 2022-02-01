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

    def will_collide(self, this_state: VehicleState, other_state: VehicleState, vehicle_body: VehicleBody) -> bool:
        """
        Check collision using circles. Return True if will collide
        
        this_state, other_state: The states of the two vehicles
        vehicle_body: The vehicle body of the two vehicles. Assuming they are the same
        """
        circles_self = v2c(this_state, vehicle_body)
        circles_other = v2c(other_state, vehicle_body)

        for circle_a, circle_b in product(circles_self, circles_other):
            dist = np.linalg.norm([circle_a[0]-circle_b[0], circle_a[1]-circle_b[1]])
            if dist < circle_a[2] + circle_b[2]:
                return True

        return False
