from typing import Dict
import numpy as np
import os
from pathlib import Path
from parksim.vehicle_types import VehicleBody
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import pandas as pd
import pygame

from parksim.pytypes import VehicleState
from parksim.utils.get_corners import get_vehicle_corners, get_vehicle_corners_from_dict

_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
# Load parking map
with open(_ROOT / 'parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)
with open(_ROOT / 'obstacles.yml') as f:
    OBSTACLE_DATA = yaml.load(f, Loader=SafeLoader).values()
MAP_SIZE = MAP_DATA['MAP_SIZE']
PARKING_AREAS = MAP_DATA['PARKING_AREAS']
WAYPOINTS = MAP_DATA['WAYPOINTS']

class InstanceCentricGenerator:
    """
    Plot the frame as semantic images
    """
    def __init__(self, spot_margin=0.3, resolution=0.1, sensing_limit=20, steps=5, stride=5):
        """
        instantiate the semantic visualizer
        
        spot_margin: the margin for seperating spot rectangles
        resolution: distance (m) per pixel. resolution = 0.1 means 0.1m per pixel
        sensing_limit: the longest distance to sense along 4 directions (m). The side length of the square = 2*sensing_limit
        steps: the number history steps to plot. If no history is desired, set the steps = 0 and stride = any value.
        stride: the stride when getting the history. stride = 1 means plot the consecutive frames. stride = 2 means plot one in every 2 frames
        """
        self.parking_spaces = self._gen_spaces()
        self.waypoints = self._gen_waypoints()

        self.map_size = MAP_SIZE

        plt.rcParams['figure.dpi'] = 125

        self.color = {'obstacle': (0, 0, 255),
                      'spot': (0, 255, 0),
                      'agent': (255, 255, 0),
                      'ego': (255, 0, 0)}
        
        self.spot_margin = spot_margin

        self.res = resolution
        self.h = int(MAP_SIZE['y'] / self.res)
        self.w = int(MAP_SIZE['x'] / self.res)

        self.sensing_limit = sensing_limit
        # 1/2 side length of the instance-centric crop. in pixel units.
        self.inst_ctr_size = int(self.sensing_limit / self.res)

        # Shrink the parking spaces a little bit
        for name in ['top_left_x', 'btm_left_x', 'btm_left_y', 'btm_right_y']:
            self.parking_spaces[name] += self.spot_margin
        for name in ['top_right_x', 'btm_right_x', 'top_left_y', 'top_right_y']:
            self.parking_spaces[name] -= self.spot_margin

        # Load the base map with drivable region
        # self.base_map = Image.open(_ROOT / 'base_map.png').convert('RGB').resize((self.w, self.h)).transpose(Image.FLIP_TOP_BOTTOM)
        pygame.init()
        self.base_map = pygame.image.load(_ROOT / 'base_map.png')
        pygame.transform.scale(self.base_map, (self.w, self.h))
        self.base_map = pygame.transform.flip(self.base_map, False, True)

        self.color = {'obstacle': (0, 0, 255),
                      'spot': (0, 255, 0),
                      'agent': (255, 255, 0),
                      'ego': (255, 0, 0)}

        self.steps = steps
        self.stride = stride

    def _get_corners(self, center, dims, angle):
        length, width = dims
        offsets = np.array([[ 0.5, -0.5],
                            [ 0.5,  0.5],
                            [-0.5,  0.5],
                            [-0.5, -0.5]])
        offsets_scaled = offsets @ np.array([[length, 0], [0, width]])

        adj_angle = np.pi - angle
        c, s = np.cos(adj_angle), np.sin(adj_angle)
        rot_mat = np.array([[c, s], [-s, c]])
        offsets_rotated = rot_mat @ offsets_scaled.T

        c = np.array([*center])
        c_stacked = np.vstack((c, c, c, c))
        return offsets_rotated.T + c_stacked

    def _gen_spaces(self):
        df = pd.DataFrame()
        idx = 0

        for ax, area in PARKING_AREAS.items():
            for a in area['areas']:
                df = df.append(self._divide_rect(a['coords'] if a['coords'] else area['bounds'], *a['shape'], idx, ax))
                idx += a['shape'][0] * a['shape'][1]

        df.columns = ['id', 'area', 'top_left_x', 'top_left_y', 'top_right_x', 'top_right_y', 'btm_right_x', 'btm_right_y', 'btm_left_x', 'btm_left_y']
        return df

    def _divide_rect(self, coords, rows, cols, start, area):
        left_x = np.linspace(coords[0][0], coords[3][0], rows + 1)
        left_y = np.linspace(coords[0][1], coords[3][1], rows + 1)

        right_x = np.linspace(coords[1][0], coords[2][0], rows + 1)
        right_y = np.linspace(coords[1][1], coords[2][1], rows + 1)

        points = np.zeros((rows + 1, cols + 1, 2))
        for i in range(rows + 1):
            x = np.linspace(left_x[i], right_x[i], cols + 1)
            y = np.linspace(left_y[i], right_y[i], cols + 1)
            points[i] = np.array(list(zip(x, y)))

        df = pd.DataFrame()
        idx = start

        for r in range(rows):
            for c in range(cols):
                df = df.append([[idx+1, area, *points[r][c], *points[r][c+1], *points[r+1][c+1], *points[r+1][c]]])
                idx += 1
                
        return df

    def _gen_waypoints(self):
        """
        generate waypoints based on yaml
        """
        waypoints = {}
        for name, segment in WAYPOINTS.items():
            bounds = segment['bounds']
            points = np.linspace(bounds[0], bounds[1], num=segment['nums'], endpoint=True)

            waypoints[name] = points

        return waypoints

    def get_history_window(self, history, num_timesteps):
        history_window = []
        index = len(history) - 1
        while index > 0 and len(history_window) < num_timesteps and len(history[index]) > 0:
            history_window.insert(0, history[index])
            index -= 1
        return history_window

    def inst_centric(self, vehicle_id, history):
        """
        crop the local region around an instance and replot it in ego color. The ego instance is always pointing towards the west

        img_frame: the image of the SAME frame
        """
        vehicle_index = vehicle_id
        NUM_STEPS = 5
        STRIDE_SIZE = 1
        instance_timeline = self.get_history_window(history, NUM_STEPS * STRIDE_SIZE)
        current_state_dict = instance_timeline[-1][vehicle_index]
        
        screen = self.plot_frame(instance_timeline)

        # Replot this specific instance with the ego color
        color_band = self._color_transition(self.color['ego'], NUM_STEPS)
        self.plot_instance_timeline(screen, color_band, instance_timeline, STRIDE_SIZE, vehicle_index)

        # The location of the instance in pixel coordinates, and the angle in degrees
        center = (np.array([current_state_dict['center-x'], current_state_dict['center-y']]) / self.res).astype('int32')
        angle_degree = current_state_dict['heading'] / np.pi * 180

        # Firstly crop a larger box which contains all rotations of the actual window
        outer_size = np.ceil(self.inst_ctr_size * np.sqrt(2))
        outer_crop_box = (center[0]-outer_size, center[1]-outer_size, center[0]+outer_size, center[1]+outer_size)

        # Rotate the larger box and crop the desired window size
        inner_crop_box = (outer_size-self.inst_ctr_size, outer_size-self.inst_ctr_size, outer_size+self.inst_ctr_size, outer_size+self.inst_ctr_size)
        print(inner_crop_box)

        outer_box = pygame.Rect(max(0, center[0] - outer_size), max(0, center[1] - outer_size), 2 * outer_size, 2 * outer_size)
        crop_outer = screen.subsurface(outer_box)
        screen.blit(crop_outer, (0, 0))

        pygame.display.update()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # quit pygame
        pygame.quit()

        # Replot this specific instance with the ego color
        color_band = self._color_transition(self.color['ego'], NUM_STEPS)
        self.plot_instance_timeline(draw, color_band, instance_timeline, STRIDE_SIZE, vehicle_index)
        #self.plot_instance(draw, self.color['ego'], ego_state)
        
        # The location of the instance in pixel coordinates, and the angle in degrees
        center = (np.array([current_state_dict['center-x'], current_state_dict['center-y']]) / self.res).astype('int32')
        angle_degree = current_state_dict['heading'] / np.pi * 180

        # Firstly crop a larger box which contains all rotations of the actual window
        outer_size = np.ceil(self.inst_ctr_size * np.sqrt(2))
        outer_crop_box = (center[0]-outer_size, center[1]-outer_size, center[0]+outer_size, center[1]+outer_size)
        
        # Rotate the larger box and crop the desired window size
        inner_crop_box = (outer_size-self.inst_ctr_size, outer_size-self.inst_ctr_size, outer_size+self.inst_ctr_size, outer_size+self.inst_ctr_size)

        img_instance = img.crop(outer_crop_box).rotate(angle_degree).crop(inner_crop_box)

        return img_instance

    def plot_frame(self, history):

        NUM_STEPS = 5
        STRIDE = 1

        # or you can use the set_mode() function to create a window and a surface at the same time
        occupy_mask = pygame.display.set_mode((self.w, self.h))

        # fill the screen with a solid color
        occupy_mask.fill((0, 0, 0)) # (R, G, B) values for white

        self.plot_obstacles(screen=occupy_mask, fill=(255, 255, 255))
        self.plot_agents(occupy_mask, (255, 255, 255), history[-1:], 1, 0)

        available = []
        for _, p in self.parking_spaces.iterrows():
            p_coords_ground = p[2:10].to_numpy().reshape((4, 2))
            p_coords_pixel = (np.array(p_coords_ground) / self.res).astype('int32')
            
            # Detect whether this spot is occupied or not
            # Only plot the spot if it is empty
            center = np.average(p_coords_pixel, axis=0).astype('int32')
            available.append(self.spot_available(occupy_mask, center, size=8))

        screen = pygame.display.set_mode((self.w, self.h))
        image = pygame.transform.scale(self.base_map, (self.w, self.h))
        screen.blit(image, (0, 0))

        self.plot_spots(available=available, screen=screen, fill=self.color['spot'])
        self.plot_obstacles(screen=screen, fill=self.color['obstacle'])
        self.plot_agents(screen, self.color['agent'], history, STRIDE, NUM_STEPS)

        return screen

    def plot_spots(self, available, screen, fill):
        """
        plot empty spots
        """
        i = 0
        for _, p in self.parking_spaces.iterrows():
            p_coords_ground = p[2:10].to_numpy().reshape((4, 2))
            p_coords_pixel = (np.array(p_coords_ground) / self.res).astype('int32')
            
            # Detect whether this spot is occupied or not
            # Only plot the spot if it is empty
            if not available[i]:
                pygame.draw.polygon(screen, fill, [tuple(p) for p in p_coords_pixel])
            i += 1

    def plot_instance(self, draw, fill, state: VehicleState):
        """
        plot a single instance at a single frame
        """
        corners_ground = get_vehicle_corners(state=state, vehicle_body=VehicleBody())
        corners_pixel = (corners_ground / self.res).astype('int32')
        draw.polygon([tuple(p) for p in corners_pixel], fill=fill)

    def plot_instance_from_state_dict(self, screen, fill, state_dict):
        """
        plot a single instance at a single frame
        """
        corners_ground = get_vehicle_corners_from_dict(state_dict)
        corners_pixel = (corners_ground / self.res).astype('int32')
        pygame.draw.polygon(screen, fill, [tuple(p) for p in corners_pixel])


    def plot_obstacles(self, screen, fill):
        """
        plot static obstacles in this scene
        """
        obstacles = OBSTACLE_DATA
        for obstacle in obstacles:
            corners_ground = self._get_corners(obstacle['coords'], obstacle['size'], obstacle['heading'])
            corners_pixel = (corners_ground / self.res).astype('int32')

            pygame.draw.polygon(screen, fill, [tuple(p) for p in corners_pixel])


    def plot_agents(self, screen, fill, history, stride, steps):
        """
        plot all moving agents and their history as fading rectangles
        """
        # Plot
        color_band = self._color_transition(fill, steps)
        for i in history[0].keys():
            self.plot_instance_timeline(screen, color_band, history, stride, i)

    def spot_available(self, occupy_mask, center, size):
        """
        detect whether a certain spot on the map is occupied or not by checking the pixel value
        center: center location (pixel) of the spot
        size: the size of the square window for occupancy detection

        return: True if empty, false if occupied
        """
        sum = 0
        for x in range(center[0]-size, center[0]+size):
            for y in range(center[1]-size, center[1]+size):
                sum += 1 if occupy_mask.get_at((x, y))[0] != 0 else 0
        
        return sum == 0
    

    def _is_visible(self, current_state, target_state):
        """
        check whether the target state is visible inside the instance-centric crop

        current_state: (x, y, heading, speed) of current instance state
        target_state: (x, y, heading, speed) of the point to be tested
        """
        theta = current_state[2]
        A = np.array([[ np.sin(theta), -np.cos(theta)], 
                      [-np.sin(theta),  np.cos(theta)], 
                      [ np.cos(theta),  np.sin(theta)], 
                      [-np.cos(theta), -np.sin(theta)]])
        b = self.sensing_limit * np.ones(4)

        offset = target_state[0:2] - current_state[0:2]

        return all( A @ offset < b)

    def global_ground_to_local_pixel(self, current_state, target_state):
        """
        transform the target state from global ground coordinates to instance-centric local crop

        current_state: numpy array (x, y, theta, velocity)
        target_state: numpy array (x, y, ...)
        """
        current_theta = current_state[2]
        R = np.array([[np.cos(-current_theta), -np.sin(-current_theta)], 
                      [np.sin(-current_theta),  np.cos(-current_theta)]])

        rotated_ground = R @ (target_state[:2] - current_state[:2])
        translation = self.sensing_limit * np.ones(2)
        translated_ground = rotated_ground + translation

        return np.floor(translated_ground / self.res).astype('int32')

    def local_pixel_to_global_ground(self, current_state, target_coords):
        """
        transform the target coordinate from pixel coordinate in the local inst-centric crop to global ground coordinates

        Note: Accuracy depends on the resolution (self.res)

        current_state: numpy array (x, y, theta, velocity)
        target_coords: numpy array (x, y) in int pixel location
        """
        scaled_local = target_coords * self.res
        translation = self.sensing_limit * np.ones(2)

        translated_local = scaled_local - translation

        current_theta = current_state[2]
        R = np.array([[np.cos(current_theta), -np.sin(current_theta)], 
                      [np.sin(current_theta),  np.cos(current_theta)]])

        rotated_local = R @ translated_local

        translated_global = rotated_local + current_state[:2]

        return translated_global

    def plot_instance_timeline(self, screen, color_band, instance_timeline, stride, agent_index):
        """
        plot the timeline of an instance
        """
        len_history = len(instance_timeline) - 1
        max_steps = np.floor( len_history / stride).astype(int)

        # History configuration
        for idx_step in range(max_steps, 0, -1):
            idx_history = len_history - idx_step * stride
            instance_state_dict = instance_timeline[idx_history][agent_index]
            self.plot_instance_from_state_dict(screen=screen, fill=color_band[-1-idx_step], state_dict=instance_state_dict)

        # Current configuration
        instance_state_dict = instance_timeline[-1][agent_index]
        self.plot_instance_from_state_dict(screen=screen, fill=color_band[-1], state_dict=instance_state_dict)

    def _color_transition(self, max_color, steps):
        """
        generate colors to plot the state history of agents

        max_color: 3-element tuple with r,g,b value. This is the color to plot the current state
        """
        # If we don't actually need the color band, return the max color directly
        if steps == 0:
            return [max_color]

        min_color = (int(max_color[0]/2), int(max_color[1]/2), int(max_color[2]/2))

        color_band = [min_color]

        for i in range(1, steps):
            r = int(min_color[0] + i * (max_color[0] - min_color[0]) / 2 / steps)
            g = int(min_color[1] + i * (max_color[1] - min_color[1]) / 2 / steps)
            b = int(min_color[2] + i * (max_color[2] - min_color[2]) / 2 / steps)
            color_band.append((r,g,b))

        color_band.append(max_color)

        return color_band

instgen = InstanceCentricGenerator()
instgen.inst_centric(1, [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {1: {'center-x': 12.63, 'center-y': 76.21, 'heading': -1.5707963267948966, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.63, 'center-y': 76.21, 'heading': -1.5707963267948966, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.63, 'center-y': 76.16, 'heading': -1.5763677764420732, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.629470715021789, 'center-y': 76.06500147444612, 'heading': -1.5819390034026675, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.627960913584522, 'center-y': 75.92950988614773, 'heading': -1.5869587064380204, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.625181913398144, 'center-y': 75.75758234426463, 'heading': -1.5910540561238138, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.621034325721112, 'center-y': 75.55286935605562, 'heading': -1.5932605391781687, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.615771863910982, 'center-y': 75.31864896707134, 'heading': -1.594350070367451, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.60962840147237, 'center-y': 75.05786977118588, 'heading': -1.5927543410007838, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.603375999391048, 'center-y': 74.773172024111, 'heading': -1.5868782314158252, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.598450489075335, 'center-y': 74.46692187525818, 'heading': -1.57804644617186, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.59608943022897, 'center-y': 74.14126965432492, 'heading': -1.56434253983013, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.598303674973, 'center-y': 73.79818209752665, 'heading': -1.5478876876154934, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.606522237488118, 'center-y': 73.439491007926, 'heading': -1.5263012784618628, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.623109265053829, 'center-y': 73.0669533805336, 'heading': -1.4425750893032565, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.672418059270763, 'center-y': 72.68450289829603, 'heading': -1.3776432648489414, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.748634353779842, 'center-y': 72.2948321398694, 'heading': -1.3248739273718644, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.847803903669316, 'center-y': 71.89973898568566, 'heading': -1.2794263336174012, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 12.967482443976696, 'center-y': 71.50068466829559, 'heading': -1.240332747570218, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.10537175290511, 'center-y': 71.09872531622005, 'heading': -1.2063058955153734, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.25953125189576, 'center-y': 70.69467796268937, 'heading': -1.1772494927064014, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.412615561109929, 'center-y': 70.32598411422411, 'heading': -1.1551191702748453, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.56173858095977, 'center-y': 69.9881412478576, 'heading': -1.1378520891549952, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.705374687019813, 'center-y': 69.67736796259828, 'heading': -1.1240847329253039, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.842805505256349, 'center-y': 69.39045947077418, 'heading': -1.1131163067758274, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 13.973736666789526, 'center-y': 69.12464319283171, 'heading': -1.104007578206313, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.098249126166696, 'center-y': 68.8775616268608, 'heading': -1.0964988359419365, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.216544045455741, 'center-y': 68.64713977929522, 'heading': -1.082064977649491, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.349466874305678, 'center-y': 68.3971719115799, 'heading': -1.0655922258015802, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.496986217573317, 'center-y': 68.13044812271653, 'heading': -1.0460385372865508, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.659472007090473, 'center-y': 67.84976626407725, 'heading': -1.016248268507334, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 14.8394966058283, 'center-y': 67.55911355171052, 'heading': -0.9876999916677488, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 15.036450144874829, 'center-y': 67.26051943650904, 'heading': -0.9596693211189304, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 15.249860066799872, 'center-y': 66.95590822235303, 'heading': -0.9312997845518962, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 15.479467755317508, 'center-y': 66.64719680410839, 'heading': -0.9016349136046293, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 15.725281066997331, 'center-y': 66.33639091718506, 'heading': -0.8696387675039512, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 15.987603548425511, 'center-y': 66.02568135690103, 'heading': -0.8342096300556175, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 16.267039074762348, 'center-y': 65.71754267227634, 'heading': -0.7049344753992588, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 16.590266632882923, 'center-y': 65.44255395295009, 'heading': -0.5891216485353943, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.63, 'center-y': 76.21, 'heading': -1.5707963267948966, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 16.94939263890804, 'center-y': 65.2025556912993, 'heading': -0.49834661621804927, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.63, 'center-y': 76.21, 'heading': -1.5707963267948966, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 17.334774352271836, 'center-y': 64.99284730498907, 'heading': -0.42434287284482647, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.63, 'center-y': 76.16, 'heading': -1.5763677764420732, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 17.740188587908214, 'center-y': 64.80968453440057, 'heading': -0.3630757722385629, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.629470715021789, 'center-y': 76.06500147444612, 'heading': -1.5819390034026675, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 18.161210603955716, 'center-y': 64.64973049095443, 'heading': -0.31255981074572275, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.627960913584522, 'center-y': 75.92950988614773, 'heading': -1.5869587064380204, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 18.594493640025192, 'center-y': 64.5097140814267, 'heading': -0.2698203840552569, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.625181913398144, 'center-y': 75.75758234426463, 'heading': -1.5910540561238138, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 19.037667368563326, 'center-y': 64.38714784030573, 'heading': -0.23365161177255017, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.621034325721112, 'center-y': 75.55286935605562, 'heading': -1.5932605391781687, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 19.488893129297157, 'center-y': 64.27975679506176, 'heading': -0.2033637384377193, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.615771863910982, 'center-y': 75.31864896707134, 'heading': -1.594350070367451, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 19.946706630074317, 'center-y': 64.18534905994979, 'heading': -0.17757485423242608, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.60962840147237, 'center-y': 75.05786977118588, 'heading': -1.5927543410007838, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 20.410006482827505, 'center-y': 64.10220286852477, 'heading': -0.15588160979621593, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.603375999391048, 'center-y': 74.773172024111, 'heading': -1.5868782314158252, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 20.877895206259204, 'center-y': 64.02867106912761, 'heading': -0.13732246254442476, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.598450489075335, 'center-y': 74.46692187525818, 'heading': -1.57804644617186, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 21.349679989347056, 'center-y': 63.96347408833151, 'heading': -0.12139301677370609, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.59608943022897, 'center-y': 74.14126965432492, 'heading': -1.56434253983013, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 21.824799126566045, 'center-y': 63.9055129526309, 'heading': -0.10797025306403293, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.598303674973, 'center-y': 73.79818209752665, 'heading': -1.5478876876154934, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 22.302776850884246, 'center-y': 63.85370409826379, 'heading': -0.09637742516246989, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.606522237488118, 'center-y': 73.439491007926, 'heading': -1.5263012784618628, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 22.783236398765744, 'center-y': 63.80725473837509, 'heading': -0.08637911326427639, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.623109265053829, 'center-y': 73.0669533805336, 'heading': -1.4425750893032565, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}, {1: {'center-x': 23.265859927537274, 'center-y': 63.76546215112172, 'heading': -0.07773711891663256, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}, 2: {'center-x': 12.672418059270763, 'center-y': 72.68450289829603, 'heading': -1.3776432648489414, 'corners': np.array([[ 2.3  ,  0.925],
       [-2.3  ,  0.925],
       [-2.3  , -0.925],
       [ 2.3  , -0.925]])}}])