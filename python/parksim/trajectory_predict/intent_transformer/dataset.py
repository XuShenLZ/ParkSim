from torch.utils.data import Dataset
import numpy as np
import torch
import os
from bisect import bisect
from abc import ABC, abstractmethod
from parksim.trajectory_predict.intent_transformer.model_utils import generate_square_subsequent_mask
_CURRENT = os.path.abspath(os.path.dirname(__file__))



class BaseTransformerDataset(ABC, Dataset, object):
    """
    We use the external dataset as a workaround for the fact the the torch
    split_dataset function returns "Subset" objects of the original dataset,
    which do not maintain the instance variables of the original. These instance
    variables are needed for model_utils to work properly, so this is a
    temporary workaround.
    """


    def __init__(self, file_paths, img_transform=None):
        """
        Instantiate the dataset
        """
        self.file_paths = file_paths
        self.img_transform = img_transform
        self.items = []

    def get_subset(self, idx):
        return Subset(self.copy_dataset(), idx)


    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        if isinstance(idx, (int, slice)): return self.items[idx]
        return [self.items[k] for k in idx]

    def copy_dataset(self):
        new_dataset = type(self)(self.file_paths, self.img_transform)
        return new_dataset

    @abstractmethod
    def process_batch_training(self, batch, device):
        ...

    @abstractmethod
    def process_batch_label(self, batch, device):
        ...

class Subset(object):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.process_batch_training = dataset.process_batch_training
        self.process_batch_label = dataset.process_batch_label

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):  
            return self.dataset[self.indices[idx]]
        else:
            return self.dataset[[self.indices[i] for i in idx]]
    
    def __len__(self):
        return len(self.indices)

class IntentTransformerDataset(BaseTransformerDataset):
    """
    Dataset containing the instance-centric crop image, the spatial traj features, intent and the label
    """
    def __init__(self, file_paths, img_transform=None):
        """
        Instantiate the dataset
        """
        super().__init__(file_paths, img_transform=None)
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
        img_history: np.ndarray = np.concatenate(all_img_history)
        trajectory_history = torch.cat(all_trajectory_history)
        intent_pose = torch.cat(all_intent_pose)
        trajectory_future = torch.cat(all_trajectory_future)
        self.items = list(zip(img_history, trajectory_history, intent_pose, trajectory_future))


        self.img_transform = img_transform


    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        
        img_history, trajectory_history, intent_pose, trajectory_future = self.items[idx]
        hist_timesteps, h, w, c = img_history.shape
        img_history_tensor = torch.empty(size=(hist_timesteps, c, h, w))
        if self.img_transform:
            for t in range(hist_timesteps):
                img_history_tensor[t] = self.img_transform(img_history[t])
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
            
        return img_history_tensor, trajectory_history, intent_pose[:,:-1], trajectory_future_tgt, trajectory_future

    def process_batch_training(self, batch, device):
        img_history, trajectory_history, intent_pose, trajectory_future_tgt, _ = batch
        img_history = img_history.to(device).float()
        trajectory_history = trajectory_history.to(device).float()
        intent_pose = intent_pose.to(device).float()
        trajectory_future_tgt = trajectory_future_tgt.to(device).float()
        mask = generate_square_subsequent_mask(trajectory_future_tgt.shape[1]).to(device).float()
        return img_history, trajectory_history, intent_pose, trajectory_future_tgt, mask

    def process_batch_label(self, batch, device):
        _, _, _, _, trajectory_future = batch
        trajectory_future = trajectory_future.to(device).float()
        return trajectory_future

class IntentTransformerV2Dataset(BaseTransformerDataset):
    """
    Dataset containing the instance-centric crop image, the spatial traj features, intent and the label
    """
    def __init__(self, file_paths, img_transform=None):
        """
        Instantiate the dataset
        """
        super().__init__(file_paths, img_transform=None)
        all_images = []
        all_trajectory_history = []
        all_intent_pose = []
        all_trajectory_future = []
        for file_path in file_paths:
            all_images.append(np.load(os.path.join(_CURRENT, f'{file_path}_image_history.npy'), mmap_mode='r')[:,-1])
            all_trajectory_history.append(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_history.npy'), mmap_mode='r'))
            all_trajectory_future.append(np.load(os.path.join(_CURRENT, f'{file_path}_trajectory_future.npy'), mmap_mode='r'))
            all_intent_pose.append(
                np.load(os.path.join(_CURRENT, f'{file_path}_intent_pose.npy'), mmap_mode='r'))
        self.start_indices = [0] * len(file_paths)
        self.data_count = 0
        for index, memmap in enumerate(all_images):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.items = list(zip(all_images, all_trajectory_history, all_intent_pose, all_trajectory_future))
        self.img_transform = img_transform



    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.data_count

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        memmap_index = bisect(self.start_indices, idx) - 1
        index_in_memmap = idx - self.start_indices[memmap_index]
        image, trajectory_history, intent_pose, trajectory_future = tuple(map(lambda x: x[index_in_memmap], self.items[memmap_index]))
        trajectory_history = torch.from_numpy(trajectory_history)
        trajectory_future = torch.from_numpy(trajectory_future)
        intent_pose = torch.from_numpy(intent_pose)
        if self.img_transform:
            image = self.img_transform(image)
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
            
        return image, trajectory_history, intent_pose[:,:-1], trajectory_future_tgt, trajectory_future

    def process_batch_training(self, batch, device):
        image, trajectory_history, intent_pose, trajectory_future_tgt, _ = batch
        image = image.to(device).float()
        trajectory_history = trajectory_history.to(device).float()
        intent_pose = intent_pose.to(device).float()
        trajectory_future_tgt = trajectory_future_tgt.to(device).float()
        mask = generate_square_subsequent_mask(trajectory_future_tgt.shape[1]).to(device).float()
        return image, trajectory_history, intent_pose, trajectory_future_tgt, mask

    def process_batch_label(self, batch, device):
        _, _, _, _, trajectory_future = batch
        trajectory_future = trajectory_future.to(device).float()
        return trajectory_future    