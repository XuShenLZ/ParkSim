from torch.utils.data import Dataset
import numpy as np
import torch

class CNNDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the non-spatial features, and the label
    """
    def __init__(self, file_path, input_transform=None, target_transform=None):
        """
        Instantiate the dataset
        """
        #self.image_features = (np.load('%s_image_feature.npy' % file_path) /
        #255).astype(np.single)
        self.image_features = np.load('%s_image_feature.npy' % file_path, mmap_mode='r+')
        #self.image_features = self.image_features.transpose(0, 3, 1, 2)
        self.non_spatial_features = np.load('%s_non_spatial_feature.npy' % file_path, mmap_mode='r+')
        self.all_labels = np.load('%s_label.npy' % file_path, mmap_mode='r+')
        self.input_transform = input_transform
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
        label = self.all_labels[idx].astype(np.single)
        if self.input_transform:
            img_feature = self.input_transform(img_feature)
            non_spatial_feature = non_spatial_feature.astype(np.single)
        if self.target_transform:
            label = self.target_transform(label)
            
        return img_feature, non_spatial_feature, label

class CNNGroupedDataset(Dataset):
    """
    Dataset containing the instance-centric crop image, the non-spatial features, and the label
    """
    def __init__(self, file_path, input_transform=None, target_transform=None):
        """
        Instantiate the dataset
        """
        #self.image_features = (np.load('%s_image_feature.npy' % file_path) /
        #255).astype(np.single)
        self.image_features = np.load('%s_image_feature_grouped_test.npy' % file_path, allow_pickle=True)
        #self.image_features = self.image_features.transpose(0, 3, 1, 2)
        self.non_spatial_features = np.load('%s_non_spatial_feature_grouped_test.npy' % file_path, allow_pickle=True)
        self.all_labels = np.load('%s_label_grouped_test.npy' % file_path, allow_pickle=True)
        self.input_transform = input_transform
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
        label = self.all_labels[idx].astype(np.single)
        if self.input_transform:
            img_feature = torch.stack([self.input_transform(feat) for feat in img_feature])
            non_spatial_feature = non_spatial_feature.astype(np.single)
        if self.target_transform:
            label = self.target_transform(label)
            
        return img_feature, non_spatial_feature, label