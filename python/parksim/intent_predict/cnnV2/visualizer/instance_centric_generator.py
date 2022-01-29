from PIL import ImageDraw, Image
from parksim.pytypes import VehicleState
from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle
import numpy as np
import os
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import pandas as pd


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
        self.base_map = Image.open(_ROOT / 'base_map.png').convert('RGB').resize((self.w, self.h)).transpose(Image.FLIP_TOP_BOTTOM)

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

    def inst_centric(self, ego_agent: RuleBasedStanleyVehicle, all_agents):
        """
        crop the local region around an instance and replot it in ego color. The ego instance is always pointing towards the west

        img_frame: the image of the SAME frame
        """
        img = self.plot_frame(all_agents)
        draw = ImageDraw.Draw(img)

        # Replot this specific instance with the ego color
        #color_band = self._color_transition(self.color['ego'], self.steps)

        #instance_timeline = self.dataset.get_agent_past(inst_token,
        #timesteps=self.steps*self.stride)
        
        self.plot_instance(draw, self.color['ego'], ego_agent)
        
        # The location of the instance in pixel coordinates, and the angle in degrees
        center = (np.array([ego_agent.state.x.x, ego_agent.state.x.y]) / self.res).astype('int32')
        angle_degree = ego_agent.state.e.psi / np.pi * 180

        # Firstly crop a larger box which contains all rotations of the actual window
        outer_size = np.ceil(self.inst_ctr_size * np.sqrt(2))
        outer_crop_box = (center[0]-outer_size, center[1]-outer_size, center[0]+outer_size, center[1]+outer_size)
        
        # Rotate the larger box and crop the desired window size
        inner_crop_box = (outer_size-self.inst_ctr_size, outer_size-self.inst_ctr_size, outer_size+self.inst_ctr_size, outer_size+self.inst_ctr_size)

        img_instance = img.crop(outer_crop_box).rotate(angle_degree).crop(inner_crop_box)

        return img_instance

    def plot_frame(self, all_agents):
        # Create the binary mask for all moving objects on the map -- static obstacles and moving agents
        occupy_mask = Image.new(mode='1', size=(self.w, self.h))
        mask_draw = ImageDraw.Draw(occupy_mask)

        # Firstly register current obstacles and agents on the binary mask
        self.plot_obstacles(draw=mask_draw, fill=1)
        self.plot_agents(mask_draw, 1, all_agents)

        img_frame = self.base_map.copy()
        img_draw = ImageDraw.Draw(img_frame)

        # Then plot everything on the main img
        self.plot_spots(occupy_mask=occupy_mask, draw=img_draw, fill=self.color['spot'])
        self.plot_obstacles(draw=img_draw, fill=self.color['obstacle'])
        self.plot_agents(img_draw, self.color['agent'], all_agents)


        return img_frame

    def plot_spots(self, occupy_mask, draw, fill):
        """
        plot empty spots
        """
        for _, p in self.parking_spaces.iterrows():
            p_coords_ground = p[2:10].to_numpy().reshape((4, 2))
            p_coords_pixel = (np.array(p_coords_ground) / self.res).astype('int32')
            
            # Detect whether this spot is occupied or not
            # Only plot the spot if it is empty
            center = np.average(p_coords_pixel, axis=0).astype('int32')
            if self.spot_available(occupy_mask, center, size=8):
                draw.polygon([tuple(p) for p in p_coords_pixel], fill=fill)

    def plot_instance(self, draw, fill, agent: RuleBasedStanleyVehicle):
        """
        plot a single instance at a single frame
        """
        corners_ground = agent.get_corners()
        corners_pixel = (corners_ground / self.res).astype('int32')
        draw.polygon([tuple(p) for p in corners_pixel], fill=fill)


    def plot_obstacles(self, draw, fill):
        """
        plot static obstacles in this scene
        """
        obstacles = OBSTACLE_DATA
        for obstacle in obstacles:
            corners_ground = self._get_corners(obstacle['coords'], obstacle['size'], obstacle['heading'])
            corners_pixel = (corners_ground / self.res).astype('int32')

            draw.polygon([tuple(p) for p in corners_pixel], fill=fill)


    def plot_agents(self, draw, fill, all_agents):
        """
        plot all moving agents and their history as fading rectangles
        """
        # Plot
        for agent in all_agents:
            self.plot_instance(draw, fill, agent)

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
                sum += occupy_mask.getpixel((x, y))
        
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