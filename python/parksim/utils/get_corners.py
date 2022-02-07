import numpy as np
from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

def get_vehicle_corners(state: VehicleState=None, vehicle_body: VehicleBody=None) -> np.ndarray:
    center = np.array([state.x.x, state.x.y])
    psi = state.e.psi

    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    corners = (R @ vehicle_body.V.T).T

    return corners + center