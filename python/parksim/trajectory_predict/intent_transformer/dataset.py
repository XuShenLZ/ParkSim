from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from torchvision import transforms
import pytorch_lightning as pl
from bisect import bisect
from parksim.trajectory_predict.intent_transformer.model_utils import generate_square_subsequent_mask, split_dataset
_CURRENT = os.path.abspath(os.path.dirname(__file__))


ALL_DATA_NUMS = ["../data/DJI_" + str(i).zfill(4) for i in range(1, 31)]

class IntentTransformerDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # called only on 1 GPU
        self.train = IntentTransformerDataset(dataset_type="train", img_transform=transforms.ToTensor())
        self.val = IntentTransformerDataset(dataset_type="val", img_transform=transforms.ToTensor())
        self.test = IntentTransformerDataset(dataset_type="test", img_transform=transforms.ToTensor())

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=256, num_workers=8, pin_memory=True, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=256, num_workers=8, pin_memory=True, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=256, num_workers=8, pin_memory=True, shuffle=False)


class IntentTransformerDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the spatial traj features, intent and the label
    """
    def __init__(self, dataset_type="train", img_transform=None):
        """
        Instantiate the dataset
        """
        super().__init__()
        self.image_history_path = os.path.join(_CURRENT, '../data/', f'{dataset_type}_image_history.npy')
        self.intent_path = os.path.join(_CURRENT, '../data/', f'{dataset_type}_intent.npy') 
        self.trajectory_history_path = os.path.join(_CURRENT, '../data/', f'{dataset_type}_trajectory_history.npy')
        self.trajectory_future_path = os.path.join(_CURRENT, '../data/', f'{dataset_type}_trajectory_future.npy')

        
        self.image_history_shape = (10, 100, 100, 3)
        self.intent_shape = (1, 3)
        self.trajectory_history_shape = (10, 3)
        self.trajectory_future_shape = (10, 3)
        
        data_sizes = np.load(os.path.join(_CURRENT, '../data/', 'dataset_sizes.npy'))

        if dataset_type == "train":
            self.data_size = data_sizes[0]
        elif dataset_type == "val":
            self.data_size = data_sizes[1]
        else:
            self.data_size = data_sizes[2]

        self.img_transform = img_transform

    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.data_size

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        img_history = np.memmap(self.image_history_path, mode='c', dtype='float32', shape=(1, *self.image_history_shape), offset=np.longlong(idx) * np.longlong(4) * np.longlong(np.prod(self.image_history_shape)))[0]
        trajectory_history = np.memmap(self.trajectory_history_path, mode='c', dtype='float32', shape=(1, *self.trajectory_history_shape), offset=np.longlong(idx) * np.longlong(4) * np.longlong(np.prod(self.trajectory_history_shape)))[0]
        intent_pose = np.memmap(self.intent_path, mode='c', dtype='float32', shape=(1, *self.intent_shape), offset=np.longlong(idx) * np.longlong(4) * np.longlong(np.prod(self.intent_shape)))[0]
        trajectory_future = np.memmap(self.trajectory_future_path, mode='c', dtype='float32', shape=(1, *self.trajectory_future_shape), offset=np.longlong(idx) * np.longlong(4) * np.longlong(np.prod(self.trajectory_future_shape)))[0]
        trajectory_history = torch.from_numpy(trajectory_history)
        trajectory_future = torch.from_numpy(trajectory_future)
        intent_pose = torch.from_numpy(intent_pose)
        hist_timesteps, h, w, c = img_history.shape
        img_history_tensor = torch.empty(size=(hist_timesteps, c, h, w))
        if self.img_transform:
            for t in range(hist_timesteps):
                img_history_tensor[t] = self.img_transform(img_history[t])
        trajectory_future_tgt = torch.cat((trajectory_history[-1:], trajectory_future[:-1])) # This is the tgt that is passed into the decoder, and trajectory_future is the label
        return img_history_tensor.float(), trajectory_history.float(), intent_pose[:,:-1].float(), trajectory_future_tgt.float(), trajectory_future.float()

class IntentTransformerV2DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # called only on 1 GPU
        self.train = IntentTransformerV2Dataset(dataset_type="train", img_transform=transforms.ToTensor())
        self.val = IntentTransformerV2Dataset(dataset_type="val", img_transform=transforms.ToTensor())
        self.test = IntentTransformerV2Dataset(dataset_type="test", img_transform=transforms.ToTensor())

    def setup(self, stage=None):
        # called on every GPU
        #all_data = IntentTransformerV2Dataset(file_paths=self.dataset_nums, img_transform=transforms.ToTensor())
        #self.train, self.val, self.test = split_dataset(all_data,
        #[self.train_proportion, self.validation_proportion,
        #self.test_proportion], split_seed=42)
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1024, num_workers=12, pin_memory=True, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1024, num_workers=12, pin_memory=True, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1024, num_workers=12, pin_memory=True, shuffle=False)

class IntentTransformerV2Dataset(IntentTransformerDataset):
    """
    Dataset containing the instance-centric crop image, the spatial traj features, intent and the label
    """
    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        img_history, trajectory_history, intent, trajectory_future_tgt, trajectory_future = super().__getitem__(idx)
            
        return img_history[-1], trajectory_history, intent, trajectory_future_tgt, trajectory_future