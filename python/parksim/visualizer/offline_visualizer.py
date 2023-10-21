import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

from parksim.pytypes import VehiclePrediction
from parksim.obstacle_types import GeofenceRegion
from parksim.vehicle_types import VehicleBody


def plot_vehicle(ax, state, steer, vehicle_body: VehicleBody):
    """
    state: array with size (3,) for x, y, psi
    steer: float for steering angle
    """
    l = vehicle_body.l
    w = vehicle_body.w

    lf = vehicle_body.lf
    lr = vehicle_body.lr

    w_d = vehicle_body.wheel_d
    w_w = vehicle_body.wheel_w

    x, y, th = state

    # Body
    p = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.array(
        [[l / 2, l / 2, -l / 2, -l / 2, l / 2], [w / 2, -w / 2, -w / 2, w / 2, w / 2]]
    )

    ax.plot(p[0, :] + x, p[1, :] + y, "k", linewidth=1)

    # Rear wheels
    p = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.array(
        [
            [w_d / 2, w_d / 2, -w_d / 2, -w_d / 2, w_d / 2],
            [w_w / 2, -w_w / 2, -w_w / 2, w_w / 2, w_w / 2],
        ]
    )

    ax.plot(
        p[0, :] + x - lr * np.cos(th) - (w - w_w) / 2 * np.sin(th),
        p[1, :] + y - lr * np.sin(th) + (w - w_w) / 2 * np.cos(th),
        "k",
        linewidth=0.5,
    )

    ax.plot(
        p[0, :] + x - lr * np.cos(th) + (w - w_w) / 2 * np.sin(th),
        p[1, :] + y - lr * np.sin(th) - (w - w_w) / 2 * np.cos(th),
        "k",
        linewidth=0.5,
    )

    # Front wheels
    p = np.array(
        [
            [np.cos(th + steer), -np.sin(th + steer)],
            [np.sin(th + steer), np.cos(th + steer)],
        ]
    ) @ np.array(
        [
            [w_d / 2, w_d / 2, -w_d / 2, -w_d / 2, w_d / 2],
            [w_w / 2, -w_w / 2, -w_w / 2, w_w / 2, w_w / 2],
        ]
    )

    ax.plot(
        p[0, :] + x + lf * np.cos(th) - (w - w_w) / 2 * np.sin(th),
        p[1, :] + y + lf * np.sin(th) + (w - w_w) / 2 * np.cos(th),
        "k",
        linewidth=0.5,
    )

    ax.plot(
        p[0, :] + x + lf * np.cos(th) + (w - w_w) / 2 * np.sin(th),
        p[1, :] + y + lf * np.sin(th) - (w - w_w) / 2 * np.cos(th),
        "k",
        linewidth=0.5,
    )

    # Arrow
    ax.arrow(
        x,
        y,
        0.2 * vehicle_body.wb * np.cos(th),
        0.2 * vehicle_body.wb * np.sin(th),
        color="black",
        width=0.04,
        head_width=0.5,
    )


class OfflineVisualizer(object):
    """
    Visualize the results offline
    """

    def __init__(
        self,
        sol: VehiclePrediction,
        obstacles,
        map,
        vehicle_body: VehicleBody,
        region: GeofenceRegion,
    ):
        self.sol = sol
        self.obstacles = obstacles
        self.map = map
        self.vehicle_body = vehicle_body
        self.region = region

    def plot_frame(self, ax, k):
        xi = self.sol.x[k]
        xj = self.sol.y[k]
        th = self.sol.psi[k]
        steer = self.sol.u_steer[k]

        plot_vehicle(ax, [xi, xj, th], steer, self.vehicle_body)

        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.plot_pyplot(ax)

        ax.set_xlim([self.region.x_min, self.region.x_max])
        ax.set_ylim([self.region.y_min, self.region.y_max])
        ax.set_aspect("equal")

    def plot_trace(self, ax):
        tick_font = {"family": "Times New Roman", "size": 20, "weight": "normal"}

        ax.plot(self.sol.t, self.sol.v, linewidth=4, label="Speed (m/s)")
        ax.plot(self.sol.t, self.sol.psi, linewidth=4, label="Heading (rad)")
        ax.plot(self.sol.t, self.sol.u_a, linewidth=4, label="Accel (m/$s^2$)")
        ax.plot(self.sol.t, self.sol.u_steer, ":", linewidth=4, label="Steer (rad)")
        ax.legend(prop={"family": "Times New Roman", "size": 15, "weight": "normal"})
        ax.set_xlabel("Time (s)", tick_font)
        ax.set_ylabel("Value", tick_font)

    def plot_trajectory(self, ax=None, step=1):
        """
        Plot the trajectory
        """

        if not ax:
            ax = plt.subplot(1, 1, 1)

        ax.plot(self.sol.x, self.sol.y, linewidth=3)

        # ax.set_xlabel("X (m)", tick_font)
        # ax.set_ylabel("Y (m)", tick_font)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        for k in np.linspace(
            0, len(self.sol.t) - 1, endpoint=True, num=int(len(self.sol.t) / step) + 1
        ):
            self.plot_frame(ax, int(k))

    def plot_solution(self, step=1, fig_path=None, show=True):
        """
        Plot solution in one figure
        step: the stride step to plot vehicle contour. 1 is every step
        """
        plt.figure()

        ax1 = plt.subplot(2, 1, 1)

        # Plot the entire trajectory
        self.plot_trajectory(ax=ax1, step=step)

        ax2 = plt.subplot(2, 1, 2)
        self.plot_trace(ax2)

        if fig_path:
            plt.savefig(fig_path)

        if show:
            plt.show(block=False)

    def animate_solution(self, interval=40, gif_path=None, show=True):
        """
        animate the solution with the specified frame interval
        interval: frame interval in millesecond
        gif_path: the path to save gif, e.g. "maneuver.gif"
        """
        fig, ax = plt.subplots()

        def animate(k):
            ax.clear()
            # Plot the entire trajectory
            ax.plot(self.sol.x, self.sol.y)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            self.plot_frame(ax, k)

        ani = FuncAnimation(
            fig, animate, frames=len(self.sol.t), interval=interval, repeat=True
        )

        if gif_path:
            writer = PillowWriter(fps=int(1000 / interval))
            ani.save(gif_path, writer=writer)

        if show:
            plt.show()

        return

    def show(self):
        plt.show()
        return
