from typing import List
import numpy as np
import dearpygui.dearpygui as dpg

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer as DlpVis

from parksim.vehicle_types import VehicleBody

class RealtimeVisualizer(object):
    """
    Realtime visualizer based on dearpy GUI for fast plotting in ROS
    """
    def __init__(self, dataset: Dataset, vehicle_body: VehicleBody, width=1900, height=1000):
        """
        width, height: the width, height of the plotting window
        """

        self.dlpvis = DlpVis(dataset)

        self.vehicle_body = vehicle_body

        dpg.create_context()
        dpg.create_viewport(title='ParkSim Simulator', width=width, height=height)

        self.map_width = self.dlpvis.map_size['x'] * 10
        self.map_height = self.dlpvis.map_size['y'] * 10

        # self.canvas = dpg.add_window(width=self.map_width, height=self.map_height, label="Map")
        self.map_window = dpg.add_window(width=self.map_width, height=self.map_height, label="Map")

        self.scene_canvas = dpg.add_draw_layer(parent=self.map_window)
        self.frame_canvas = dpg.add_draw_layer(parent=self.map_window)

        with dpg.theme() as win_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255,255,255,255))

        dpg.bind_item_theme(self.map_window, win_theme)

        # Plot lines and static obstacles
        self.draw_scene(scene_token=dataset.list_scenes()[0])

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def __del__(self):
        dpg.destroy_context()

    def _xy2p(self, x, y):
        """
        x, y: ground coordinates in (m)
        convert x, y to pixel coordinates
        """
        px = x * 10
        py = (self.dlpvis.map_size['y'] - y) * 10
        return px, py

    def _draw_waypoints(self):
        """ 
        plot the waypoints on the map as scatters
        """
        for _, points in self.dlpvis.waypoints.items():
            pxs, pys = self._xy2p(points[:,0], points[:, 1])

            for px, py in zip(pxs, pys):
                dpg.draw_circle([px, py], 2, fill=(0,0,0,128), parent=self.scene_canvas)

    def _draw_parking_lines(self):
        """
        draw parking lines
        """
        for _, p in self.dlpvis.parking_spaces.iterrows():
            p_coords = p[2:10].to_numpy().reshape((4, 2))

            px, py = self._xy2p(p_coords[:,0], p_coords[:, 1])

            dpg.draw_quad(p1=[px[0], py[0]], p2=[px[1], py[1]], p3=[px[2], py[2]], p4=[px[3], py[3]], color=(0,0,0,255), parent=self.scene_canvas)

    def _draw_obstacles(self, scene_token):
        """
        plot static obstacles in this scene
        """
        scene = self.dlpvis.dataset.get('scene', scene_token)
        for obstacle_token in scene['obstacles']:
            obstacle = self.dlpvis.dataset.get('obstacle', obstacle_token)
            corners = self.dlpvis._get_corners(obstacle['coords'], obstacle['size'], obstacle['heading'])

            px, py = self._xy2p(corners[:,0], corners[:, 1])

            dpg.draw_quad(p1=[px[0], py[0]], p2=[px[1], py[1]], p3=[px[2], py[2]], p4=[px[3], py[3]], fill=(0,0,255,255), color=(0,0,0,0), parent=self.scene_canvas)

    def _gen_vehicle_corners(self, state):
        """
        state: array-like with size (3,) for x, y, psi
        """
        l = self.vehicle_body.l
        w = self.vehicle_body.w

        x, y, th = state

        # Body
        p = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]]) @ np.array([[l/2,l/2, -l/2, -l/2, l/2],[w/2, -w/2, -w/2, w/2, w/2]])
        
        p[0, :] += x
        p[1, :] += y
        
        return p.T

    def draw_line(self, points: np.ndarray, color: tuple):
        """
        draw a (curved) line with a list of x, y coordinates

        points: np.array with size Nx2
        color: (RGBA) tuple
        """
        poli_xy = []
        for x, y in points:
            px, py = self._xy2p(x, y)
            poli_xy.append([px, py])

        dpg.draw_polyline(points=poli_xy, color=color, parent=self.frame_canvas)

    def draw_scene(self, scene_token):
        """
        plot lines and static obstacles in a specified scene
        """

        self._draw_waypoints()

        # Plot parking lines
        self._draw_parking_lines()

        # Plot static obstacles
        self._draw_obstacles(scene_token)

    def clear_frame(self):
        dpg.delete_item(self.frame_canvas, children_only=True)

    def draw_vehicle(self, states: List[np.ndarray], fill=(255,128,0,255)):
        """
        states: an array-like object with size 3 that contains the (x, y, psi) states in (m, rad)
        color: (r, g, b, a) tuple in range 0-255
        """
        corners = self._gen_vehicle_corners(states)

        px, py = self._xy2p(corners[:,0], corners[:, 1])
        dpg.draw_quad(p1=[px[0], py[0]], p2=[px[1], py[1]], p3=[px[2], py[2]], p4=[px[3], py[3]], fill=fill, color=(0,0,0,0), parent=self.frame_canvas)

    def draw_frame(self, frame_token):
        frame = self.dlpvis.dataset.get('frame', frame_token)
        
        # Plot instances
        for inst_token in frame['instances']:
            instance = self.dlpvis.dataset.get('instance', inst_token)
            agent = self.dlpvis.dataset.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self.dlpvis._get_corners(instance['coords'], agent['size'], instance['heading'])

                px, py = self._xy2p(corners[:,0], corners[:, 1])

                dpg.draw_quad(p1=[px[0], py[0]], p2=[px[1], py[1]], p3=[px[2], py[2]], p4=[px[3], py[3]], fill=(255,128,0,255), color=(0,0,0,0), parent=self.frame_canvas)

    def render(self):
        """
        render a frame after updating contents
        """
        if dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()


if __name__ == "__main__":

    from pathlib import Path

    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    ds.load(home_path + '/dlp-dataset/data/DJI_0012')

    vehicle_body = VehicleBody()

    vis = RealtimeVisualizer(ds, vehicle_body)

    scene = ds.get('scene', ds.list_scenes()[0])

    states_list = [[10,10,0], [50, 70, np.pi/2], [70, 15, -np.pi/2]]

    frames = ds.get_future_frames(scene['first_frame'], 100)

    while dpg.is_dearpygui_running():
        for frame in frames:
            vis.clear_frame()
            # vis.draw_frame(frame["frame_token"])
            for states in states_list:
                vis.draw_vehicle(states)
            dpg.render_dearpygui_frame()
