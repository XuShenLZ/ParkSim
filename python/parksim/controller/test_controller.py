from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from parksim.controller.mpc_controller import MPC
from parksim.controller_types import MPCParams
from parksim.path_planner.offline_maneuver import OfflineManeuver

from pathlib import Path

from parksim.pytypes import VehiclePrediction, VehicleState
from tqdm import tqdm

offline_maneuver_path = "/ParkSim/data/parking_maneuvers.pickle"

offline_maneuver = OfflineManeuver(str(Path.home()) + offline_maneuver_path)

path = offline_maneuver.get_maneuver(
    driving_dir="east", x_position="left", spot="north", heading="up"
)

path.x += 10
path.y += 10

state = VehicleState()
state.t = 0
state.x.x = path.x[0]
state.x.y = path.y[0]
state.e.psi = path.psi[0]
state.v.v = path.v[0]

state.u.u_a = 0
state.u.u_steer = 0

mpc = MPC()
mpc.setup(P=np.diag([1, 1, 0, 0]), Q=np.diag([1, 1, 0, 0]), R=np.zeros((2, 2)))

state_hist = [state.copy()]

obs_vertices = {0: np.array([[-5, -5], [-5, -4], [-4, -4], [-4, -5]])}


plt.ion()
fig = plt.figure()
plt.axis("equal")
plt.plot(path.x, path.y)


(pred_line,) = plt.plot([], [], "o", lw=2)
(ref_line,) = plt.plot([], [], "^", lw=2)

for _ in tqdm(range(100)):
    tspan = np.linspace(state.t, state.t + mpc.N * mpc.dt, num=mpc.N, endpoint=False)

    ref = VehiclePrediction()
    ref.x = np.interp(tspan, path.t, path.x)
    ref.y = np.interp(tspan, path.t, path.y)
    ref.psi = np.interp(tspan, path.t, path.psi)
    ref.v = np.interp(tspan, path.t, path.v)

    pred = mpc.step(state=state, ref=ref, obstacle_corners=obs_vertices)
    state.t += mpc.dt

    state_hist.append(state.copy())

    pred_line.set_xdata(pred.x)
    pred_line.set_ydata(pred.y)

    ref_line.set_xdata(ref.x)
    ref_line.set_ydata(ref.y)

    fig.canvas.draw()
    fig.canvas.flush_events()

    sleep(0.5)


all_x = []
all_y = []
for state in state_hist:
    all_x.append(state.x.x)
    all_y.append(state.x.y)

plt.plot(all_x, all_y)

# plt.show()
