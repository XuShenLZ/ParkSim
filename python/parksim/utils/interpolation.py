import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from parksim.pytypes import VehiclePrediction

def interpolate_states_inputs(states: VehiclePrediction, new_t: np.ndarray):
    result = states.copy()

    result.t = new_t

    result.x = np.interp(result.t, states.t, states.x)
    result.y = np.interp(result.t, states.t, states.y)
    result.v = np.interp(result.t, states.t, states.v)

    sparse_time, sparse_psi = remove_close_timesteps(states.t, states.psi)
    old_r = R.from_euler('z', sparse_psi, degrees=False)
    slerp = Slerp(sparse_time, old_r)

    result.psi = slerp(new_t).as_euler('xyz', degrees=False)[:, 2]

    result.u_a = np.interp(result.t, states.t, states.u_a)
    result.u_steer = np.interp(result.t, states.t, states.u_steer)

    return result

def remove_close_timesteps(times, values, tol=0.001):
    sparse_times = [times[0]]
    sparse_values = [values[0]]

    for i in range(1, len(times)):
        if times[i] - times[i-1] > tol:
            sparse_times.append(times[i])
            sparse_values.append(values[i])

    return sparse_times, sparse_values