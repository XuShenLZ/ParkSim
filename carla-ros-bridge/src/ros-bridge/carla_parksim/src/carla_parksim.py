#!/usr/bin/env python

from __future__ import print_function

import datetime
import math
from threading import Thread

import numpy
from transforms3d.euler import quat2euler
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_msgs.msg import CarlaStatus
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.msg import CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleControl
from carla_msgs.msg import CarlaLaneInvasionEvent
from carla_msgs.msg import CarlaCollisionEvent
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool

# ==============================================================================
# -- CarlaParkSim --------------------------------------------------------------
# ==============================================================================


class CarlaParkSim(CompatibleNode):

    """
    Handles the spawning of the ego vehicle and its sensors

    Derive from this class and implement method sensors()
    """

    def __init__(self):
        super(CarlaParkSim, self).__init__('carla_parksim')
        self._surface = None
        self.host = self.get_param('host', 'localhost')
        self.port = self.get_param('port', 2000)

        self.camera_name = self.get_param("camera_name", "camera_1")

        self.image_subscriber = self.new_subscription(
            Image, "/carla/{}/image".format(self.camera_name),
            self.on_view_image, qos_profile=10)
        self.loginfo('parksim node intitialized')

    def on_collision(self, data):
        """
        Callback on collision event
        """
        intensity = math.sqrt(data.normal_impulse.x**2 +
                              data.normal_impulse.y**2 + data.normal_impulse.z**2)
        self.hud.notification('Collision with {} (impulse {})'.format(
            data.other_actor_id, intensity))

    def on_lane_invasion(self, data):
        """
        Callback on lane invasion event
        """
        text = []
        for marking in data.crossed_lane_markings:
            if marking is CarlaLaneInvasionEvent.LANE_MARKING_OTHER:
                text.append("Other")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                text.append("Broken")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                text.append("Solid")
            else:
                text.append("Unknown ")
        self.hud.notification('Crossed line %s' % ' and '.join(text))

    def on_view_image(self, image):
        """
        Callback when receiving a camera image
        """
        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, game_clock, display):
        """
        render the current image
        """

        # do_quit = self.controller.parse_events(game_clock)
        # if do_quit:
        #     return
        # self.hud.tick(game_clock)

        if self._surface is not None:
            display.blit(self._surface, (0, 0))
        # self.hud.render(display)

    


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main(args=None):
    """
    main function
    """
    roscomp.init("parksim", args=args)

    resolution = {"width": 800, "height": 600}

    pygame.init()
    pygame.display.set_caption("CARLA ParkSim")

    try:
        display = pygame.display.set_mode((resolution['width'], resolution['height']),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        parksim_node = CarlaParkSim()
        clock = pygame.time.Clock()

        executor = roscomp.executors.MultiThreadedExecutor()
        executor.add_node(parksim_node)

        spin_thread = Thread(target=parksim_node.spin)
        spin_thread.start()

        roscomp.loginfo("inside parksim main()")

        while roscomp.ok():
            clock.tick_busy_loop(60)
            if parksim_node.render(clock, display):
                return
            pygame.display.flip()
    except KeyboardInterrupt:
        roscomp.loginfo("User requested shut down.")
    finally:
        roscomp.shutdown()
        spin_thread.join()
        pygame.quit()


if __name__ == '__main__':
    main()
