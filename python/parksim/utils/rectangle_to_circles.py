import numpy as np

import matplotlib.pyplot as plt

from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody
from parksim.obstacle_types import RectangleObstacle

def v2c(state: VehicleState, vehicle_body: VehicleBody):
    """
    Use a few circles to approximate vehicle body rectangle
    num_circles: the number of circles to approximate
    """
    radius = vehicle_body.w/2

    start_offset = vehicle_body.cr
    start_xc = state.x.x - start_offset * np.cos(state.e.psi)
    start_yc = state.x.y - start_offset * np.sin(state.e.psi)

    end_offset = vehicle_body.cf
    end_xc = state.x.x + end_offset * np.cos(state.e.psi)
    end_yc = state.x.y + end_offset * np.sin(state.e.psi)

    xcs = np.linspace(start_xc, end_xc, vehicle_body.num_circles, endpoint=True)
    ycs = np.linspace(start_yc, end_yc, vehicle_body.num_circles, endpoint=True)

    circles = []
    for xc, yc in zip(xcs, ycs):
        circles.append((xc, yc, radius))

    return circles


def main():
    state = VehicleState()
    state.x.x = 1
    state.x.y = 2
    state.e.psi = np.pi/4
    vehicle_body = VehicleBody()

    fig, ax = plt.subplots(1)

    rect = RectangleObstacle(xc=state.x.x, yc=state.x.y, w=vehicle_body.l, h=vehicle_body.w, psi=state.e.psi)
    
    rect.plot_pyplot(ax)

    circles = v2c(state, vehicle_body)

    for circle in circles:
        cir = plt.Circle((circle[0], circle[1]), circle[2], color='b')
        ax.add_patch(cir)

    plt.show()

if __name__ == "__main__":
    main()