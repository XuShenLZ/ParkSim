from abc import abstractmethod, ABC

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

class AbstractAgent(ABC):
    """
    Abstract Agent Class
    """
    def __init__(self, state: VehicleState, vehicle_body: VehicleBody):
        self.state = state
        self.vehicle_body = vehicle_body

