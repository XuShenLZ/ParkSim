from itertools import product
import matplotlib.pyplot as plt
from parksim.obstacle_types import GeofenceRegion, RectangleObstacle
from parksim.path_planner.offline_maneuver import OfflineManeuver
from parksim.pytypes import VehiclePrediction
from parksim.vehicle_types import VehicleBody
from parksim.visualizer.offline_visualizer import OfflineVisualizer


def main():
    vehicle_body = VehicleBody(vehicle_flag=0)
    region = GeofenceRegion(x_max=6.5, x_min=-6.5, y_max=9, y_min=-9)

    obstacles = [
        RectangleObstacle(xc=-3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=-6.11, w=5, h=5.22),
        RectangleObstacle(xc=-3.8, yc=6.11, w=5, h=5.22),
        RectangleObstacle(xc=3.8, yc=6.11, w=5, h=5.22),
    ]

    offline_maneuver = OfflineManeuver(pickle_file="./data/parking_maneuvers.pickle")

    plt.figure(figsize=(16, 10))

    driving_dir = "east"

    i = 1
    for x_pos, spot, heading in product(
        ["left", "right"], ["north", "south"], ["up", "down"]
    ):
        traj: VehiclePrediction = offline_maneuver.get_maneuver(
            driving_dir=driving_dir, x_position=x_pos, spot=spot, heading=heading
        )

        vis: OfflineVisualizer = OfflineVisualizer(
            sol=traj,
            obstacles=obstacles,
            map=None,
            vehicle_body=vehicle_body,
            region=region,
        )

        ax = plt.subplot(2, 4, i)

        vis.plot_trajectory(ax=ax, step=int(len(traj.x) / 2))

        ax.set_title(
            f"{x_pos}-{spot}-{heading}",
            fontdict={"family": "Times New Roman", "size": 30},
        )

        i += 1

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.15)
    plt.tight_layout()
    plt.savefig("offline_maneuvers.pdf")

    # plt.figure(figsize=(16, 10))

    traj: VehiclePrediction = offline_maneuver.get_maneuver(
        driving_dir="east", x_position="left", spot="north", heading="up"
    )
    vis: OfflineVisualizer = OfflineVisualizer(
        sol=traj,
        obstacles=obstacles,
        map=None,
        vehicle_body=vehicle_body,
        region=region,
    )

    f, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 3], figsize=(10, 4))
    vis.plot_trajectory(ax1, step=int(len(traj.x) / 2))

    plt.yticks(fontname="Times New Roman", fontsize=17)
    plt.xticks(fontname="Times New Roman", fontsize=17)
    # plt.xlabel(fontname="Times New Roman", fontsize=15)
    # plt.ylabel(fontname="Times New Roman", fontsize=15)

    vis.plot_trace(ax2)

    plt.tight_layout()
    plt.savefig("state_input_profile.pdf")

    plt.show()


if __name__ == "__main__":
    main()
