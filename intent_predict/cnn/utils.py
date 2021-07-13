from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    """
    Dataset containing the instance-centric crop and the intent heatmap
    """
    def __init__(self, file_path, transform=None, target_transform=None):
        """
        Instantiate the dataset
        """
        self.all_feature = np.load('%s_feature.npy' % file_path)
        self.all_label = np.load('%s_label.npy' % file_path)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.all_label.shape[0]

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        feature = self.all_feature[idx]
        label = self.all_label[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        return feature, label