from dataclasses import dataclass, field
from typing import List
from casadi.casadi import Function
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from parksim.controller.dynamic_model import kinematic_bicycle_rk

from parksim.pytypes import PythonMsg, VehicleState, VehiclePrediction
from parksim.vehicle_types import VehicleBody, VehicleConfig
from parksim.obstacle_types import GeofenceRegion, RectangleObstacle

from parksim.path_planner.hybrid_astar.hybrid_a_star import (
    hybrid_a_star_planning,
    hybrid_a_star_plotting,
)

from parksim.visualizer.offline_visualizer import OfflineVisualizer


@dataclass
class PlannerConfig(PythonMsg):
    N: int = field(default=50)

    Q: np.ndarray = field(default=np.eye(4) * 0.01)
    R: np.ndarray = field(default=np.eye(2) * 0.1)

    x0: VehicleState = field(default=None)
    xf: VehicleState = field(default=None)

    # For hybrid A*
    xy_resolution: float = field(default=1.0)
    yaw_resolution: float = field(default=np.deg2rad(10.0))

    dmin: float = field(default=0.001)


class HobcaPlanner:
    """
    HOBCA Planner
    """

    def __init__(
        self,
        config: PlannerConfig,
        vehicle_body: VehicleBody,
        vehicle_config: VehicleConfig,
        region: GeofenceRegion,
    ):

        assert isinstance(config, PlannerConfig)
        assert isinstance(vehicle_body, VehicleBody)
        assert isinstance(vehicle_config, VehicleConfig)
        assert isinstance(region, GeofenceRegion)

        self.config = config
        self.vehicle_body = vehicle_body
        self.vehicle_config = vehicle_config
        self.region = region

        self._setup_bicycle_model()

    def warm_start_state(
        self, x0: VehicleState, xf: VehicleState, obstacles: List[RectangleObstacle]
    ):
        print("Warm Start States with Hybrid A* planning")

        obs_grid_size = self.vehicle_body.w / 4.0

        ox, oy = [], []

        for obs in obstacles:
            for i in range(4):
                num_steps = int(
                    np.linalg.norm(obs.xy[i] - obs.xy[i + 1]) / obs_grid_size
                )
                for x, y in zip(
                    np.linspace(obs.xy[i, 0], obs.xy[i + 1, 0], num_steps),
                    np.linspace(obs.xy[i, 1], obs.xy[i + 1, 1], num_steps),
                ):
                    ox.append(x)
                    oy.append(y)

        start = [x0.x.x, x0.x.y, x0.q.to_yaw()]
        goal = [xf.x.x, xf.x.y, xf.q.to_yaw()]

        ws_path = hybrid_a_star_planning(
            start=start,
            goal=goal,
            ox=ox,
            oy=oy,
            xy_resolution=self.config.xy_resolution,
            yaw_resolution=self.config.yaw_resolution,
        )

        hybrid_a_star_plotting(start, goal, ws_path, ox, oy)

        if ws_path == ([], [], []):
            print("Hybrid A* Failed")
            return 0, None
        else:
            print("Hybrid A* Succeed")

            N = len(ws_path.x_list) - 1

            ws_traj = VehiclePrediction()
            ws_traj.x = ws_path.x_list
            ws_traj.y = ws_path.y_list
            ws_traj.psi = ws_path.yaw_list
            ws_traj.v = [0] * (N + 1)

            ws_traj.u_a = [0] * N
            ws_traj.u_steer = [0] * N

            return N, ws_traj

    def solve_ws(
        self, N: int, obstacles: List[RectangleObstacle], ws_traj: VehiclePrediction
    ):
        print("Solving Dual WS Problem...")

        n_obs = len(obstacles)

        n_hps = []
        for obs in obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        opti_ws = ca.Opti()

        l = opti_ws.variable(sum(n_hps), N)
        m = opti_ws.variable(4 * n_obs, N)
        d = opti_ws.variable(n_obs, N)

        obj = 0

        opti_ws.subject_to(ca.vec(l) >= 0)
        opti_ws.subject_to(ca.vec(m) >= 0)

        for k in range(N):
            t = np.array([ws_traj.x[k], ws_traj.y[k]])
            R = np.array(
                [
                    [np.cos(ws_traj.psi[k]), -np.sin(ws_traj.psi[k])],
                    [np.sin(ws_traj.psi[k]), np.cos(ws_traj.psi[k])],
                ]
            )

            for j, obs in enumerate(obstacles):
                idx0 = sum(n_hps[:j])
                idx1 = sum(n_hps[: j + 1])
                lj = l[idx0:idx1, k]
                mj = m[4 * j : 4 * (j + 1), k]

                opti_ws.subject_to(
                    ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj) == d[j, k]
                )
                opti_ws.subject_to(veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2))
                opti_ws.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) <= 1)

                obj -= d[j, k]

        opti_ws.minimize(obj)

        p_opts = {"expand": True}
        s_opts = {"print_level": 0}
        opti_ws.solver("ipopt", p_opts, s_opts)

        sol = opti_ws.solve()
        print(sol.stats()["return_status"])

        return sol.value(l), sol.value(m)

    def solve(
        self,
        N: int,
        x0: VehicleState,
        xf: VehicleState,
        obstacles: List[RectangleObstacle],
        ws_traj: VehiclePrediction,
        ws_l,
        ws_m,
    ):
        print("Solving Main Problem...")

        state_u = np.array(
            [self.region.x_max, self.region.y_max, np.inf, self.vehicle_config.v_max]
        ).T
        state_l = np.array(
            [self.region.x_min, self.region.y_min, -np.inf, self.vehicle_config.v_min]
        ).T

        input_u = np.array([self.vehicle_config.a_max_parking, self.vehicle_config.delta_max]).T
        input_l = np.array([self.vehicle_config.a_min, self.vehicle_config.delta_min]).T

        n_obs = len(obstacles)

        n_hps = []
        for obs in obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        ws_z = np.zeros((4, N + 1))
        ws_z[0, :] = ws_traj.x
        ws_z[1, :] = ws_traj.y
        ws_z[2, :] = ws_traj.psi
        ws_z[3, :] = ws_traj.v

        z0 = np.array([x0.x.x, x0.x.y, x0.q.to_yaw(), x0.v.mag()]).T
        zf = np.array([xf.x.x, xf.x.y, xf.q.to_yaw(), xf.v.mag()]).T

        opti = ca.Opti()

        l = opti.variable(sum(n_hps), N)
        m = opti.variable(4 * n_obs, N)
        z = opti.variable(4, N + 1)
        u = opti.variable(2, N)

        obj = 0

        opti.subject_to(ca.vec(l) >= 0)
        opti.subject_to(ca.vec(m) >= 0)

        opti.subject_to(z[:, 0] == z0)
        opti.subject_to(z[:, N] == zf)

        opti.subject_to(opti.bounded(state_l, z[:, N], state_u))

        for k in range(N):

            opti.subject_to(opti.bounded(state_l, z[:, k], state_u))
            opti.subject_to(opti.bounded(input_l, u[:, k], input_u))

            opti.subject_to(z[:, k + 1] == self.f(z[:, k], u[:, k]))

            t = z[:2, k]
            yaw = z[2, k]
            R = ca.vertcat(
                ca.horzcat(ca.cos(yaw), -ca.sin(yaw)),
                ca.horzcat(ca.sin(yaw), ca.cos(yaw)),
            )

            for j, obs in enumerate(obstacles):
                idx0 = sum(n_hps[:j])
                idx1 = sum(n_hps[: j + 1])
                lj = l[idx0:idx1, k]
                mj = m[4 * j : 4 * (j + 1), k]

                opti.subject_to(
                    ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj)
                    >= self.config.dmin
                )
                opti.subject_to(veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2))
                opti.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) == 1)

            obj += ca.bilin(
                self.config.Q, z[:, k] - ws_z[:, k], z[:, k] - ws_z[:, k]
            ) + ca.bilin(self.config.R, u[:, k], u[:, k])

        opti.minimize(obj)

        p_opts = {"expand": True}
        s_opts = {
            "print_level": 3,
            "tol": 1e-2,
            "constr_viol_tol": 1e-3,
            "max_iter": 300,
        }
        opti.solver("ipopt", p_opts, s_opts)

        opti.set_initial(z, ws_z)
        opti.set_initial(m, ws_m)
        opti.set_initial(l, ws_l)

        sol = opti.solve()

        result = VehiclePrediction()
        result.t = np.linspace(0, N * self.vehicle_config.dt, N + 1)
        result.x = sol.value(z)[0, :]
        result.y = sol.value(z)[1, :]
        result.psi = sol.value(z)[2, :]
        result.v = sol.value(z)[3, :]

        result.u_a = np.append(sol.value(u)[0, :], sol.value(u)[0, -1])
        result.u_steer = np.append(sol.value(u)[1, :], sol.value(u)[1, -1])

        return result

    def _setup_bicycle_model(self):
        dt = self.vehicle_config.dt
        M = self.vehicle_config.M
        self.f = kinematic_bicycle_rk(dt, self.vehicle_body, M)


def main():
    config = PlannerConfig()
    vehicle_body = VehicleBody(vehicle_flag=0)
    vehicle_config = VehicleConfig()
    region = GeofenceRegion(x_max=8, x_min=-8, y_max=11, y_min=-11)

    obstacles = [
        RectangleObstacle(xc=-3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=-3.8, yc=6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=6.11, w=5, h=5.22),
    ]

    planner = HobcaPlanner(
        config=config,
        vehicle_body=vehicle_body,
        vehicle_config=vehicle_config,
        region=region,
    )

    init_state = VehicleState()
    final_state = VehicleState()

    init_state.x.x = -4.0
    init_state.x.y = -1.75  # 4.35 or 7.85
    init_state.q.from_yaw(0)

    final_state.x.x = 0.0
    final_state.x.y = 6.11
    final_state.q.from_yaw(np.pi / 2)

    N, ws_traj = planner.warm_start_state(
        x0=init_state, xf=final_state, obstacles=obstacles
    )

    ws_l, ws_m = planner.solve_ws(N=N, obstacles=obstacles, ws_traj=ws_traj)

    opt_traj = planner.solve(
        N=N,
        x0=init_state,
        xf=final_state,
        obstacles=obstacles,
        ws_traj=ws_traj,
        ws_l=ws_l,
        ws_m=ws_m,
    )

    vis = OfflineVisualizer(
        sol=opt_traj,
        obstacles=obstacles,
        map=None,
        vehicle_body=vehicle_body,
        region=region,
    )

    vis.plot_solution(step=50)
    vis.animate_solution(interval=int(1000 * vehicle_config.dt))


from itertools import product
import pickle


def datagen():
    config = PlannerConfig()
    vehicle_body = VehicleBody(vehicle_flag=0)
    vehicle_config = VehicleConfig()
    region = GeofenceRegion(x_max=8, x_min=-8, y_max=11, y_min=-11)

    obstacles = [
        RectangleObstacle(xc=-3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=-3.8, yc=6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=6.11, w=5, h=5.22),
    ]

    planner = HobcaPlanner(
        config=config,
        vehicle_body=vehicle_body,
        vehicle_config=vehicle_config,
        region=region,
    )

    start_x = {"left": -4.0, "right": 4.0}
    end_y = {"north": 6.11, "south": -6.11}
    end_psi = {"up": np.pi / 2, "down": -np.pi / 2}

    result = {}

    for sx, ey, ep in product(start_x, end_y, end_psi):
        init_state = VehicleState()
        final_state = VehicleState()

        init_state.x.x = start_x[sx]
        init_state.x.y = -1.75
        init_state.q.from_yaw(0)

        final_state.x.x = 0.0
        final_state.x.y = end_y[ey]
        final_state.q.from_yaw(end_psi[ep])

        N, ws_traj = planner.warm_start_state(
            x0=init_state, xf=final_state, obstacles=obstacles
        )

        ws_l, ws_m = planner.solve_ws(N=N, obstacles=obstacles, ws_traj=ws_traj)

        opt_traj = planner.solve(
            N=N,
            x0=init_state,
            xf=final_state,
            obstacles=obstacles,
            ws_traj=ws_traj,
            ws_l=ws_l,
            ws_m=ws_m,
        )

        vis = OfflineVisualizer(
            sol=opt_traj,
            obstacles=obstacles,
            map=None,
            vehicle_body=vehicle_body,
            region=region,
        )

        vis.plot_solution(step=50, fig_path="%s_%s_%s.png" % (sx, ey, ep), show=False)
        vis.animate_solution(
            interval=int(1000 * vehicle_config.dt),
            gif_path="%s_%s_%s.gif" % (sx, ey, ep),
            show=False,
        )

        key = ("east", sx, ey, ep)
        result[key] = np.array(
            [
                opt_traj.t,
                opt_traj.x,
                opt_traj.y,
                opt_traj.psi,
                opt_traj.v,
                opt_traj.u_a,
                opt_traj.u_steer,
            ]
        )
        print(key, result[key].shape)

    # Mirror the direction
    m_x = {"left": "right", "right": "left"}
    m_y = {"north": "south", "south": "north"}
    m_p = {"up": "down", "down": "up"}

    for sx, ey, ep in product(start_x, end_y, end_psi):
        original = result[("east", sx, ey, ep)].copy()
        original[1:3, :] *= -1
        original[3, :] = (original[3, :] + 2 * np.pi) % (2 * np.pi) - np.pi
        result[("west", m_x[sx], m_y[ey], m_p[ep])] = original

    with open("parking_maneuvers.pickle", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
    # datagen()
