import numpy as np

from dlp.visualizer import SemanticVisualizer
from dlp.dataset import Dataset

from PIL import Image, ImageDraw

class PostProcessor(object):
    """
    Post processing the data for learning
    """
    def __init__(self, dataset: Dataset, steps=10, stride=5):
        """
        instantiation
        """
        self.vis = SemanticVisualizer(dataset, steps=steps, stride=stride)

    def _is_inside(self, current_state, target_state):
        """
        current_state: (x, y, heading, speed) of current instance state
        target_state: (x, y, heading, speed) of the point to be tested
        """
        theta = current_state[2]
        A = np.array([[ np.sin(theta), -np.cos(theta)], 
                      [-np.sin(theta),  np.cos(theta)], 
                      [ np.cos(theta),  np.sin(theta)], 
                      [-np.cos(theta), -np.sin(theta)]])
        b = self.vis.sensing_limit * np.ones(4)

        offset = target_state[0:2] - current_state[0:2]

        return all( A @ offset < b)

    def global_ground_to_local_pixel(self, current_state, label_ground):
        """
        transform the label coordinate from global ground coordinates to instance-centric local crop
        """
        current_theta = current_state[2]
        R = np.array([[np.cos(-current_theta), -np.sin(-current_theta)], 
                      [np.sin(-current_theta),  np.cos(-current_theta)]])

        rotated_ground = R @ (label_ground[:2] - current_state[:2])
        translation = self.vis.sensing_limit * np.ones(2)
        translated_ground = rotated_ground + translation

        return np.floor(translated_ground / self.vis.res).astype('int32')


    def compute_keypoint(self, inst_token):
        """
        compute the intent keypoint for this instance. The keypoint will be a pixel location on the instance-centric crop image
        """
        traj = self.vis.dataset.get_future_traj(inst_token)
        current_state = traj[0]

        # Check whether the final point of the trajetory is inside the window
        if self._is_inside(current_state, traj[-1]):
            # If the final point is inside the window (instance-centric crop)
            label_ground = traj[-1, 0:2]
            label_pixel = self.global_ground_to_local_pixel(current_state, label_ground)
        else:
            # If the final point is outside the window
            # Find the first index that goes outside the window
            first_outside_idx = 1
            while self._is_inside(current_state, traj[first_outside_idx]):
                first_outside_idx += 1

            # The label pixel is the pixel corresponds to the last inside point
            label_ground = traj[first_outside_idx-1, 0:2]
            label_pixel = self.global_ground_to_local_pixel(current_state, label_ground)

        return label_pixel

    def gen_label(self, keypoint: np.ndarray, stride=4, sigma=2):
        """
        generate a 3d low-resolution label array with 3-channels: heatmap (1) and xy-offset (2,3)
        range of values in the array: [0, 1]

        keypoint: 2-element np array
        stride: image stride from input size to output heatmap
        """
        height = np.floor(self.vis.inst_ctr_size*2 / stride).astype('int32')
        width = height

        label = np.zeros((width, height, 3))

        p = np.floor(keypoint / stride).astype('int32')
        offset = keypoint / stride - p

        for i in range(height):
            for j in range(width):
                label[i, j, 0] = np.exp( -((i-p[1])**2 + (j-p[0])**2) / 2 / sigma**2)

        label[p[1], p[0], 1] = offset[0]
        label[p[1], p[0], 2] = offset[1]

        return label

    def gen_feature_label(self, inst_token, img_frame, stride=4, sigma=2):
        """
        generate the feature and label pair
        """
        img_instance = self.vis.inst_centric(img_frame, inst_token)

        keypoint = self.compute_keypoint(inst_token)
        label = self.gen_label(keypoint, stride=stride, sigma=sigma)

        return img_instance, label
        
