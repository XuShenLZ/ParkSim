# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import pygame
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import pdb
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np
import random


from carla_utils import *
# if sys.version_info >= (3, 0):
#     from configparser import ConfigParser
# else:
#     from ConfigParser import RawConfigParser as ConfigParser
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    drone_camera = None

    spwnr = None
    steady_spwnr = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_moode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter)

        blueprint_library = world.world.get_blueprint_library()
        drone_camera_bp = blueprint_library.find('sensor.camera.rgb')
        drone_camera_bp.set_attribute('image_size_x', str(600))
        drone_camera_bp.set_attribute('image_size_y', str(800))
        drone_camera_bp.set_attribute('fov', '100')
        drone_camera_bp.set_attribute('sensor_tick', '0.1')
        drone_camera_transform = carla.Transform(carla.Location(x=298.0, y=20.0, z=50.0), carla.Rotation(yaw=154, pitch=-90))
        drone_camera = world.world.spawn_actor(drone_camera_bp, drone_camera_transform)


    except Exception as e:
        print('go an exception', e)
    finally:
        if args.record:
            client.stop_recorder()
        if drone_camera:
            drone_camera.destroy()
        if world is not None:
            # time.sleep(f)
            steady_spwnr.remove()
            spwnr.remove()
            # extra_spwnr.remove()
            world.destroy()
        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')


    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user.')



if __name__ == '__main__':
    main()