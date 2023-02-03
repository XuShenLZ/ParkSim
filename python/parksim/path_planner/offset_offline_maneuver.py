import pickle
import matplotlib.pyplot as plt
import random

from parksim.pytypes import VehiclePrediction

random.seed(0)

class OffsetOfflineManeuver(object):
    """
    Library of offline maneuver
    """
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as handle:
            self.lib = pickle.load(handle)

    def get_maneuver(self, xy_offset=[0,0], 
                        offset=0.0,
                        driving_dir=random.choice(['east', 'west']), 
                        x_position=random.choice(['left', 'right']),
                        spot=random.choice(['north', 'south']),
                        heading=None) -> VehiclePrediction:
        # print('Trajectory requested:', (driving_dir, x_position, spot, heading))
        
        if heading is None:
            if (offset, driving_dir, x_position, spot, 'up') in self.lib:
                traj = self.lib[(offset, driving_dir, x_position, spot, 'up')]
            else:
                traj = self.lib[(offset, driving_dir, x_position, spot, 'down')]

        res = VehiclePrediction()
        res.t = traj[:, 0]
        res.x = traj[:, 1] + xy_offset[0]
        res.y = traj[:, 2] + xy_offset[1]
        res.psi = traj[:, 3]
        res.v = traj[:, 4]

        # res.u_a = traj[5, :]
        # res.u_steer = traj[6, :]

        return res

def main():
    offline_maneuver = OffsetOfflineManeuver(pickle_file='offset_parking_maneuvers.pickle')
    state, input = offline_maneuver.get_maneuver()

    plt.figure()

    ax1 = plt.subplot(2,1,1)

    # Plot the entire trajectory
    ax1.plot(state['x'], state['y'])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')

    ax2 = plt.subplot(2,1,2)
    ax2.plot(state['t'], state['v'])
    ax2.plot(state['t'], state['yaw'])
    ax2.plot(state['t'], input['a'])
    ax2.plot(state['t'], input['steer'], ':')
    ax2.legend(('speed','angle','accel','steer'))
    ax2.set_xlabel('time')

    plt.show()

if __name__ == "__main__":
    main()