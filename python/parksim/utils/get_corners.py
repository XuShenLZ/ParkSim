import numpy as np
from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody

def get_vehicle_corners(state: VehicleState=None, vehicle_body: VehicleBody=None) -> np.ndarray:
    
    state_dict = {}
    state_dict['center-x'] = state.x.x
    state_dict['center-y'] = state.x.y
    state_dict['heading'] = state.e.psi
    state_dict['corners'] = vehicle_body.V

    return get_vehicle_corners_from_dict(state_dict)

def get_vehicle_corners_from_dict(state_dict):
    x = state_dict['center-x']
    y = state_dict['center-y']
    psi = state_dict['heading']
    body_shape = state_dict['corners']
    center = np.array([x, y])

    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    corners = (R @ body_shape.T).T

    return corners + center