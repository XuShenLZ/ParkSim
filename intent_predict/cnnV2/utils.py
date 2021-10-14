from torch.utils.data import Dataset
import numpy as np

class CNNDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the non-spatial features, and the label
    """
    def __init__(self, file_path, transform=None, target_transform=None):
        """
        Instantiate the dataset
        """
        self.image_features = np.load('%s_image_feature.npy' % file_path)
        self.non_spatial_features = np.load('%s_non_spatial_feature.npy' % file_path)
        self.all_labels = np.load('%s_label.npy' % file_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Overwrite the get length method for dataset
        """
        return self.all_labels.shape[0]

    def __getitem__(self, idx):
        """
        Overwrite the get item method for dataset
        """
        img_feature = self.image_features[idx]
        non_spatial_feature = self.non_spatial_features[idx]
        label = self.all_labels[idx]
        if self.transform:
            img_feature = self.transform(img_feature)
            non_spatial_feature = self.transform(non_spatial_feature)
        if self.target_transform:
            label = self.target_transform(label)
            
        return img_feature, non_spatial_feature, label