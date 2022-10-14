from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
from .network import SmallRegularizedCNN
from torch.utils.data import DataLoader
from torchvision import transforms

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


def get_predictions(model, dji_path):
    model.eval().cuda()
    dataset = CNNGroupedDataset(dji_path, input_transform = transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, num_workers=12)
    data_size = len(dataset)
    running_top_1_accuracy = 0
    running_top_3_accuracy = 0
    running_top_5_accuracy = 0
    for batch in tqdm(dataloader):
        all_img_features, all_non_spatial_features, all_labels = batch
        for img_feature, non_spatial_feature, labels in zip(all_img_features, all_non_spatial_features, all_labels):
            img_feature = img_feature.cuda()
            non_spatial_feature = non_spatial_feature.cuda()
            labels = labels.cuda()
            num_options = labels.shape[0]
            #model.forward(img_feature, non_spatial_feature)
            #inputs, labels = data[0].to(device), data[1].to(device)

            #optimizer.zero_grad()

            preds = model(img_feature, non_spatial_feature)
            labels = labels.unsqueeze(1)
            label = torch.argmax(labels)
            preds = preds.flatten()
            preds = torch.topk(preds, min(5, num_options))
            pred_indices = preds.indices
            if label in pred_indices[:1]:
                running_top_1_accuracy += 1 / data_size
            if label in pred_indices[:min(3, num_options)]:
                running_top_3_accuracy += 1 / data_size
            if label in pred_indices[:min(5, num_options)]:
                running_top_5_accuracy += 1 / data_size

            #print(label, pred_indices)
            
    #print(f"Top 1 Accuracy: {running_top_1_accuracy}")
    #print(f"Top 3 Accuracy: {running_top_3_accuracy}")
    #print(f"Top 5 Accuracy: {running_top_5_accuracy}")
    return running_top_1_accuracy, running_top_3_accuracy, running_top_5_accuracy