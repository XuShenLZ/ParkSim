import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

class MGCVAEDataModule(pl.LightningDataModule):
    # TODO: update default batch size
    def __init__(self, num_workers=8, batch_size = 10) -> None:
        super().__init__()
        self.num_workers=num_workers
        self.batch_size = batch_size
    
    def prepare_data(self) -> None:
        self.train = MGCVAEDataset(dataset_type="train", img_transform=None)
        self.val = MGCVAEDataset(dataset_type="val", img_transform=None)
        self.test = MGCVAEDataset(dataset_type="test", img_transform=None)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

class MGCVAEDataset(Dataset):
    """
    Dataset containing ego vehicle state history, target vehicle state history, sum of neighboring vehicle/pedestrian state histories, and target vehicle state future
    """
    def __init__(self, dataset_type="train", img_transform=None) -> None:
        super().__init__()
        # TODO: figure out what these numbers should be
        # (num(history), dim(state))
        # for now, ego vehicle = target vehicle
        self.ego_history_shape = (10, 4)
        self.target_history_shape = (10,4)
        self.neighbor_veh_history_shape = (10, 4)
        self.neighbor_ped_history_shape = (10, 2)
        self.target_future_shape = (10, 4)
        # TODO: are these the map dims?
        self.map_shape = (3, 400, 400)

        # TODO: update to read from an actual data file
        if dataset_type == "train":
            self.data_size = 380
        elif dataset_type == "val":
            self.data_size = 10
        else:
            self.data_size = 10
        
        # TODO: update to read from data file
        self.ego_history = np.load("/home/nidhi/MPCLab/ParkSim/python/parksim/trajectronpp/data/all_e_saved.npy").transpose(2,1,0)[:, :10, :]
        self.target_history = np.load("/home/nidhi/MPCLab/ParkSim/python/parksim/trajectronpp/data/all_e_saved.npy").transpose(2,1,0)[:, :10, :]
        self.neighbor_veh_history = np.load("/home/nidhi/MPCLab/ParkSim/python/parksim/trajectronpp/data/all_v_saved.npy").transpose(2,1,0)[:, :10, :]
        print(f"{self.neighbor_veh_history.shape=}")
        self.neighbor_ped_history = np.load("/home/nidhi/MPCLab/ParkSim/python/parksim/trajectronpp/data/all_p_saved.npy").transpose(2,1,0)[:, :10, :]
        self.target_future = np.load("/home/nidhi/MPCLab/ParkSim/python/parksim/trajectronpp/data/all_e_saved.npy").transpose(2,1,0)[:, 10:, :2]
        self.map = np.random.rand(self.data_size, *self.map_shape)
        
        # TODO: is img transform necessary?
        self.img_transform = img_transform
    
    def __len__(self):
        """
        Overwrite default dataset len method
        """
        return self.data_size

    def __getitem__(self, idx):
        # TODO: maybe set all the histories to a single tensor so that there aren't so many vars?
        ego_history = self.ego_history[idx]
        target_history = self.target_history[idx]
        neighbor_veh_history = self.neighbor_veh_history[idx]
        neighbor_ped_history = self.neighbor_ped_history[idx]
        target_future = self.target_future[idx]
        map = self.map[idx]

        ego_history = torch.from_numpy(ego_history)
        target_history = torch.from_numpy(target_history)
        neighbor_veh_history = torch.from_numpy(neighbor_veh_history)
        neighbor_ped_history = torch.from_numpy(neighbor_ped_history)
        target_future = torch.from_numpy(target_future)
        map = torch.from_numpy(map)
        return ego_history.float(), target_history.float(), neighbor_veh_history.float(), neighbor_ped_history.float(), target_future.float(), map.float()
