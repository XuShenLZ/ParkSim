from typing import Dict, List, Tuple
import casadi as ca
import numpy as np
from parksim.controller_types import MPCParams
from parksim.obstacle_types import GeofenceRegion
from parksim.pytypes import VehiclePrediction, VehicleState
from parksim.utils.get_corners import rectangle_to_polytope
from parksim.vehicle_types import VehicleBody, VehicleConfig
from parksim.controller.dynamic_model import kinematic_bicycle_rk


class MPC(object):
    """
    MPC Controller
    """

    def __init__(
        self,
        control_params: MPCParams = MPCParams(),
        region: GeofenceRegion = GeofenceRegion(),
        vehicle_body: VehicleBody = VehicleBody(),
        vehicle_config: VehicleConfig = VehicleConfig(),
    ):
        """
        init
        """
        super().__init__()

        self.N = control_params.N
        self.dt = control_params.dt

        self.Q = control_params.Q
        self.R = control_params.R

        self.obs_buffer_size = control_params.obs_buffer_size

        self.static_dist = control_params.static_distance
        self.static_radius = control_params.static_radius

        self.region = region

        self.vehicle_body = vehicle_body
        self.vehicle_config = vehicle_config

        self.back_up_steps: int = 0

        self.pred: VehiclePrediction = None

    def setup(self):
        """
        setup the controller
        """
        self.opti = ca.Opti()

        f_dt = kinematic_bicycle_rk(dt=self.dt, vehicle_body=self.vehicle_body)
        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        # Decision variables
        self.x = self.opti.variable(self.N)
        self.y = self.opti.variable(self.N)
        self.psi = self.opti.variable(self.N)
        self.v = self.opti.variable(self.N)

        self.a = self.opti.variable(self.N)
        self.delta = self.opti.variable(self.N)

        # Assume all obstacles are rectangular
        self.l = self.opti.variable(self.N, 4 * self.obs_buffer_size)
        self.m = self.opti.variable(self.N, 4 * self.obs_buffer_size)

        # Parameters for references
        self.ref_x = self.opti.parameter(self.N)
        self.ref_y = self.opti.parameter(self.N)
        self.ref_psi = self.opti.parameter(self.N)
        self.ref_v = self.opti.parameter(self.N)

        # Parameters for initial constraints
        self.x0 = self.opti.parameter()
        self.y0 = self.opti.parameter()
        self.psi0 = self.opti.parameter()
        self.v0 = self.opti.parameter()

        # Parameters for obstacles
        self.obstacles = []
        for _ in range(self.obs_buffer_size):
            self.obstacles.append(
                {"A": self.opti.parameter(4, 2), "b": self.opti.parameter(4)}
            )

        # Initial constraints
        self.opti.subject_to(self.x[0] == self.x0)
        self.opti.subject_to(self.y[0] == self.y0)
        self.opti.subject_to(self.psi[0] == self.psi0)
        self.opti.subject_to(self.v[0] == self.v0)

        self.opti.subject_to(self.l[:] >= 0)
        self.opti.subject_to(self.m[:] >= 0)

        J = 0

        for k in range(self.N):
            # State and input range
            self.opti.subject_to(
                self.opti.bounded(self.region.x_min, self.x[k], self.region.x_max)
            )

            self.opti.subject_to(
                self.opti.bounded(self.region.y_min, self.y[k], self.region.y_max)
            )

            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.v_min, self.v[k], self.vehicle_config.v_max
                )
            )

            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.a_min, self.a[k], self.vehicle_config.a_max_parking
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.delta_min,
                    self.delta[k],
                    self.vehicle_config.delta_max,
                )
            )

            # Dynamics constraints and stage cost
            if k < self.N - 1:
                state = ca.vertcat(
                    self.x[k],
                    self.y[k],
                    self.psi[k],
                    self.v[k],
                )
                input = ca.vertcat(self.a[k], self.delta[k])
                state_p = ca.vertcat(
                    self.x[k + 1],
                    self.y[k + 1],
                    self.psi[k + 1],
                    self.v[k + 1],
                )

                self.opti.subject_to(state_p == f_dt(state, input))

            J += (
                self.Q[0] * (self.x[k] - self.ref_x[k]) ** 2
                + self.Q[1] * (self.y[k] - self.ref_y[k]) ** 2
                + self.Q[2] * (self.psi[k] - self.ref_psi[k]) ** 2
                + self.Q[3] * (self.v[k] - self.ref_v[k]) ** 2
                + self.R[0] * self.a[k] ** 2
                + self.R[1] * self.delta[k] ** 2
            )

            # OBCA constraints
            t = ca.vertcat(self.x[k], self.y[k])
            R = ca.vertcat(
                ca.horzcat(ca.cos(self.psi[k]), -ca.sin(self.psi[k])),
                ca.horzcat(ca.sin(self.psi[k]), ca.cos(self.psi[k])),
            )
            for j, obs in enumerate(self.obstacles):
                lj = self.l[k, 4 * j : 4 * (j + 1)].T
                mj = self.m[k, 4 * j : 4 * (j + 1)].T

                self.opti.subject_to(
                    ca.dot(-veh_g, mj) + ca.dot((obs["A"] @ t - obs["b"]), lj)
                    >= self.static_dist
                )
                self.opti.subject_to(
                    veh_G.T @ mj + R.T @ obs["A"].T @ lj == np.zeros(2)
                )
                self.opti.subject_to(ca.dot(obs["A"].T @ lj, obs["A"].T @ lj) == 1)

        self.opti.minimize(J)

        p_opts = {
            "expand": True,
            "print_time": False,
        }
        s_opts = {
            "print_level": 0,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            "max_iter": 600,
            # "mumps_mem_percent": 64000,
            # "linear_solver": "ma97",
            # "ma97": {"print_level": -1},
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def _adv_onestep(self, array: np.ndarray):
        """
        advance the array one step forward
        """
        if len(array.shape) == 1:
            result = np.append(array[1:], array[-1])
        elif len(array.shape) == 2:
            result = np.vstack([array[1:, :], array[-1, :]])
        else:
            raise ValueError(
                "unexpected shape when advancing the array to one step ahead."
            )

        return result

    def step(
        self,
        state: VehicleState,
        ref: VehiclePrediction,
        obstacle_corners: Dict[Tuple, np.ndarray] = None,
        obstacle_As: List[np.ndarray] = None,
        obstacle_bs: List[np.ndarray] = None,
    ):
        """
        step the controller with the current state and obstacles
        """
        assert (obstacle_corners is None) or (
            obstacle_As is None and obstacle_bs is None
        ), "can only pass in obstacle_corners OR obstacle_As and obstacle_bs"

        # Initial state
        self.opti.set_value(self.x0, state.x.x)
        self.opti.set_value(self.y0, state.x.y)
        self.opti.set_value(self.psi0, state.e.psi)
        self.opti.set_value(self.v0, state.v.v)

        # references
        self.opti.set_value(self.ref_x, ref.x)
        self.opti.set_value(self.ref_y, ref.y)
        self.opti.set_value(self.ref_psi, ref.psi)
        self.opti.set_value(self.ref_v, ref.v)

        # Fill in the obstacle dimensions
        idx = 0
        if obstacle_corners is not None:
            for v in obstacle_corners.values():
                if any(
                    [
                        np.linalg.norm([c[0] - state.x.x, c[1] - state.x.y])
                        < self.static_radius
                        for c in v
                    ]
                ):
                    assert (
                        idx < self.obs_buffer_size
                    ), "the number of obstacles exceeds the buffer size. Consider enlarge the buffer or shrink the static_radius"

                    A, b = rectangle_to_polytope(v)
                    self.opti.set_value(self.obstacles[idx]["A"], A)
                    self.opti.set_value(self.obstacles[idx]["b"], b)
                    idx += 1
        else:
            for A, b in zip(obstacle_As, obstacle_bs):
                assert (
                    idx < self.obs_buffer_size
                ), "the number of obstacles exceeds the buffer size. Consider enlarge the buffer or shrink the static_radius"

                self.opti.set_value(self.obstacles[idx]["A"], A)
                self.opti.set_value(self.obstacles[idx]["b"], b)
                idx += 1

        # The rest of obstacles in the buffer would be filled with nonsense value
        for _idx in range(idx, self.obs_buffer_size):
            self.opti.set_value(
                self.obstacles[_idx]["A"],
                np.array(
                    [
                        [1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1],
                    ]
                ),
            )
            self.opti.set_value(
                self.obstacles[_idx]["b"],
                np.array(
                    [
                        self.region.x_min + 1,
                        -self.region.x_min,
                        self.region.y_min + 1,
                        -self.region.y_min,
                    ]
                ),
            )

        # At the first iteration, initialize pred with reference
        if self.pred is None:
            self.pred = ref.copy()
            self.pred.u_a = np.zeros(self.N)
            self.pred.u_steer = np.zeros(self.N)

        # initial guess
        self.opti.set_initial(self.x, self._adv_onestep(self.pred.x))
        self.opti.set_initial(self.y, self._adv_onestep(self.pred.y))
        self.opti.set_initial(self.psi, self._adv_onestep(self.pred.psi))
        self.opti.set_initial(self.v, self._adv_onestep(self.pred.v))

        self.opti.set_initial(self.a, self._adv_onestep(self.pred.u_a))
        self.opti.set_initial(self.delta, self._adv_onestep(self.pred.u_steer))

        try:
            sol = self.opti.solve()
            # print(sol.stats()["return_status"]
            self.back_up_steps = self.N - 1

            self.pred.x = sol.value(self.x)
            self.pred.y = sol.value(self.y)
            self.pred.psi = sol.value(self.psi)
            self.pred.v = sol.value(self.v)

            self.pred.u_a = sol.value(self.a)
            self.pred.u_steer = sol.value(self.delta)

        except:
            print(
                f"=== Solving failed, {self.back_up_steps} remaining backup steps. ====="
            )
            self.back_up_steps -= 1

            self.pred.x = self._adv_onestep(self.pred.x)
            self.pred.y = self._adv_onestep(self.pred.y)
            self.pred.psi = self._adv_onestep(self.pred.psi)
            self.pred.v = self._adv_onestep(self.pred.v)

            self.pred.u_a = self._adv_onestep(self.pred.u_a)
            self.pred.u_steer = self._adv_onestep(self.pred.u_steer)

        # Update the current state as well
        state.x.x = self.pred.x[1]
        state.x.y = self.pred.y[1]
        state.e.psi = self.pred.psi[1]
        state.v.v = self.pred.v[1]

        state.u.u_a = self.pred.u_a[0]
        state.u.u_steer = self.pred.u_steer[0]

        return self.pred
