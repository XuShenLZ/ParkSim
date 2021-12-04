"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

This code originally comes from PythonRobotics Repository https://github.com/AtsushiSakai/PythonRobotics

Modified by Xu Shen (@XuShenLZ)

"""

from math import sqrt, cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from parksim.vehicle_types import VehicleBody, VehicleConfig

vehicle_body = VehicleBody()
vehicle_config = VehicleConfig()

WB = vehicle_body.lf + vehicle_body.lr  # rear to front wheel
W = vehicle_body.w  # width of car
# LB = (vehicle_body.l - vehicle_body.lr - vehicle_body.lf) / 2  # distance from rear to vehicle back end
LB = vehicle_body.l / 2 # distance from center to rear end
# LF = WB + LB  # distance from rear to vehicle front end
LF = LB # distance from center to front end
MAX_STEER = vehicle_config.delta_max  # [rad] maximum steering angle

W_BUBBLE_R = sqrt((vehicle_body.l / 2.0) ** 2 + 1)

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x
        cy = i_y

        ids = kd_tree.query_ball_point([cx, cy], W_BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # no collision

    return True  # collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


def plot_car(x, y, yaw):
    car_color = '-k'
    c, s = cos(yaw), sin(yaw)
    rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    arrow_x, arrow_y, arrow_yaw = x, y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    # yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2
    yaw += pi_2_pi(distance * sin(steer) / LB)

    return x, y, yaw


def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_car(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()
