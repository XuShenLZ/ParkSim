import numpy as np

from dlp.visualizer import SemanticVisualizer
from dlp.dataset import Dataset
from torch.utils import data

class KptProcessor(object):
    """
    Compute keypoints from semantic image
    """
    def __init__(self, dataset: Dataset, vis_steps=10, vis_stride=5):
        self.vis = SemanticVisualizer(dataset, steps=vis_steps, stride=vis_stride)

    def compute_keypoint(self, inst_token):
        """
        compute the intent keypoint for this instance. The keypoint will be a pixel location on the instance-centric crop image
        """
        traj = self.vis.dataset.get_future_traj(inst_token)
        current_state = traj[0]

        # Check whether the final point of the trajetory is inside the window
        if self.vis._is_visible(current_state, traj[-1]):
            # If the final point is inside the window (instance-centric crop)
            label_ground = traj[-1, 0:2]
            label_pixel = self.vis.global_ground_to_local_pixel(current_state, label_ground)
        else:
            # If the final point is outside the window
            # Find the first index that goes outside the window
            first_outside_idx = 1
            while self.vis._is_visible(current_state, traj[first_outside_idx]):
                first_outside_idx += 1

            # The label pixel is the pixel corresponds to the last inside point
            label_ground = traj[first_outside_idx-1, 0:2]
            label_pixel = self.vis.global_ground_to_local_pixel(current_state, label_ground)

        return label_pixel

    def gen_feature_label(self, inst_token, img_frame):
        """
        generate the feature and label pair
        """
        img_instance = self.vis.inst_centric(img_frame, inst_token)

        keypoint = self.compute_keypoint(inst_token)

        return np.asarray(img_instance), keypoint

class ImgProcessor(KptProcessor):
    """
    Generate image labels
    """
    def __init__(self, dataset: Dataset, vis_steps=10, vis_stride=5):
        """
        instantiation
        """
        super().__init__(dataset=dataset, vis_steps=vis_steps, vis_stride=vis_stride)

    def gen_img_label(self, keypoint: np.ndarray, img_stride=4, sigma=2):
        """
        generate a 3d low-resolution label array with 3-channels: heatmap (1) and xy-offset (2,3)
        range of values in the array: [0, 1]

        keypoint: 2-element np array
        img_stride: image stride from input size to output heatmap
        """
        height = np.floor(self.vis.inst_ctr_size*2 / img_stride).astype('int32')
        width = height

        label = np.zeros((width, height, 3))

        p = np.floor(keypoint / img_stride).astype('int32')
        offset = keypoint / img_stride - p

        for i in range(height):
            for j in range(width):
                label[i, j, 0] = np.exp( -((i-p[1])**2 + (j-p[0])**2) / 2 / sigma**2)

        label[p[1], p[0], 1] = offset[0]
        label[p[1], p[0], 2] = offset[1]

        return label

    def gen_feature_label(self, inst_token, img_frame, img_stride=4, sigma=2):
        """
        generate the feature and label pair
        """
        img_instance = self.vis.inst_centric(img_frame, inst_token)

        keypoint = self.compute_keypoint(inst_token)
        img_label = self.gen_img_label(keypoint, img_stride=img_stride, sigma=sigma)

        return np.asarray(img_instance), img_label
        
