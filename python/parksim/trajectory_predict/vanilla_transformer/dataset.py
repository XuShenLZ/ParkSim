from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import os

_CURRENT = os.path.abspath(os.path.dirname(__file__))

class CNNTransformerDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the non-spatial features, and the label
    """
    def __init__(self, file_path, img_transform=None):
        """
        Instantiate the dataset
        """
        #self.image_features = (np.load('%s_image_feature.npy' % file_path) /
        #255).astype(np.single)
        image_features_pil = np.load('%s_image_feature.npy' % file_path, allow_pickle=True) #Data is stored as PIL images right now :(
        self.image_features = np.array([np.asarray(img) for img in image_features_pil])
        self.trajectory_history = torch.from_numpy(np.load('%s_trajectory_history.npy' % file_path))
        self.trajectory_future = torch.from_numpy(np.load('%s_trajectory_future.npy' % file_path))

        combined_data = torch.cat((self.trajectory_history, self.trajectory_future))
        self.max_norm = torch.max(torch.norm(combined_data, dim=2))

        self.trajectory_history /= self.max_norm
        self.trajectory_future /= self.max_norm

        #self.trajectory_history = torch.nn.functional.normalize(self.trajectory_history, dim=2)
        #self.trajectory_future = torch.nn.functional.normalize(self.trajectory_future, dim=2)
        self.img_transform = img_transform


    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.image_features.shape[0]

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        img_feature = self.image_features[idx]
        trajectory_history = self.trajectory_history[idx]
        trajectory_future = self.trajectory_future[idx]
        if self.img_transform:
            img_feature = self.img_transform(img_feature)
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
            
        return img_feature, trajectory_history, trajectory_future_tgt, trajectory_future


class CNNTransformerDatasetMulti(Dataset):
    """
    Dataset containing the instance-centric crop image, the non-spatial features, and the label
    """
    def __init__(self, file_paths, img_transform=None):
        """
        Instantiate the dataset
        """
        #self.image_features = (np.load('%s_image_feature.npy' % file_path) /
        #255).astype(np.single)

        all_img_history = []
        all_trajectory_history = []
        all_trajectory_future = []
        for file_path in file_paths:
            all_img_history.append(np.load(os.path.join(_CURRENT, f'{file_path}_image_history.npy')))
            all_trajectory_history.append(torch.from_numpy(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_history.npy'))))
            all_trajectory_future.append(torch.from_numpy(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_future.npy'))))
        self.img_history: np.ndarray = np.concatenate(all_img_history)
        self.trajectory_history = torch.cat(all_trajectory_history)
        self.trajectory_future = torch.cat(all_trajectory_future)

        #combined_data = torch.cat((self.trajectory_history, self.trajectory_future)).clone()

        #self.min = torch.amin(combined_data, dim=(0,1))
        #self.max = torch.amax(combined_data, dim=(0,1))

        #self.trajectory_history = (self.trajectory_history - self.min) / (self.max - self.min)
        #self.trajectory_future  = (self.trajectory_future - self.min) / (self.max - self.min)

        #self.max_norm = torch.max(torch.norm(self.combined_data, dim=2))

        #self.trajectory_history /= self.max_norm
        #self.trajectory_future /= self.max_norm

        #self.trajectory_history = torch.nn.functional.normalize(self.trajectory_history, dim=2)
        #self.trajectory_future = torch.nn.functional.normalize(self.trajectory_future, dim=2)
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
        trajectory_future = self.trajectory_future[idx]

        hist_timesteps, h, w, c = img_history.shape
        img_history_tensor = torch.empty(size=(hist_timesteps, c, h, w))
        if self.img_transform:
            for t in range(hist_timesteps):
                img_history_tensor[t] = self.img_transform(img_history[t])
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
            
        return img_history_tensor, trajectory_history, trajectory_future_tgt, trajectory_future
