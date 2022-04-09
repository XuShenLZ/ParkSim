from torch.utils.data import Dataset
import numpy as np
import torch
import os

_CURRENT = os.path.abspath(os.path.dirname(__file__))

class IntentTransformerDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the spatial traj features, intent and the label
    """
    def __init__(self, file_paths, img_transform=None):
        """
        Instantiate the dataset
        """
        all_img_history = []
        all_trajectory_history = []
        all_intent_pose = []
        all_trajectory_future = []
        for file_path in file_paths:

            all_img_history.append(np.load(os.path.join(_CURRENT, f'{file_path}_image_history.npy')))
            all_trajectory_history.append(torch.from_numpy(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_history.npy'))))
            all_trajectory_future.append(torch.from_numpy(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_future.npy'))))
            all_intent_pose.append(torch.from_numpy(
                np.load(os.path.join(_CURRENT, f'{file_path}_intent_pose.npy'))))
        self.img_history: np.ndarray = np.concatenate(all_img_history)
        self.trajectory_history = torch.cat(all_trajectory_history)
        self.intent_pose = torch.cat(all_intent_pose)
        self.trajectory_future = torch.cat(all_trajectory_future)

        self.img_transform = img_transform


    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.img_history.shape[0]

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        img_history = self.img_history[idx]
        trajectory_history = self.trajectory_history[idx]
        intent_pose = self.intent_pose[idx][0,:-1]
        trajectory_future = self.trajectory_future[idx]

        hist_timesteps, h, w, c = img_history.shape
        img_history_tensor = torch.empty(size=(hist_timesteps, c, h, w))
        if self.img_transform:
            for t in range(hist_timesteps):
                img_history_tensor[t] = self.img_transform(img_history[t])
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
            
        return img_history_tensor, trajectory_history, intent_pose, trajectory_future_tgt, trajectory_future
