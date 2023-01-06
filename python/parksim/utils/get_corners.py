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

def rectangle_to_polytope(corners):
    A = []
    b = []
    if corners[0][0] == corners[1][0]: # facing up/down
        if corners[0][1] < corners[1][1]: # facing down
            A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b = [corners[0][0], -corners[2][0], corners[1][1], -corners[3][1]]
        else: # facing up
            A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b = [corners[3][0], -corners[0][0], corners[0][1], -corners[1][1]]
    elif corners[0][1] == corners[1][1]: # facing left/right
        if corners[0][0] < corners[1][0]: # facing left
            A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b = [corners[1][0], -corners[0][0], corners[2][1], -corners[0][1]]
        else: # facing right
            A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b = [corners[0][0], -corners[1][0], corners[0][1], -corners[2][1]]
    else: # rotated
        coefs = [np.polynomial.polynomial.Polynomial.fit([corners[i][0], corners[(i + 1) % 4][0]], [corners[i][1], corners[(i + 1) % 4][1]], 1).convert().coef for i in range(4)]
        if corners[0][0] < corners[1][0]: # left edge on bottom
            A.extend([[coefs[0][1], -1], [-coefs[2][1], 1]])
            b.extend([-coefs[0][0], coefs[2][0]])
        else:
            A.extend([[-coefs[0][1], 1], [coefs[2][1], -1]])
            b.extend([coefs[0][0], -coefs[2][0]])
        if corners[1][0] < corners[2][0]: # bottom edge on bottom
            A.extend([[coefs[1][1], -1], [-coefs[3][1], 1]])
            b.extend([-coefs[1][0], coefs[3][0]])
        else:
            A.extend([[-coefs[1][1], 1], [coefs[3][1], -1]])
            b.extend([coefs[1][0], -coefs[3][0]])

    return np.array(A), np.array(b)