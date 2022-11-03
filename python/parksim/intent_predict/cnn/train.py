from parksim.intent_predict.cnn.utils import CNNDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
from tqdm import tqdm
from .utils import get_predictions

import os
from datetime import datetime
from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.intent_predict.cnn.pytorchtools import EarlyStopping


_CURRENT = os.path.abspath(os.path.dirname(__file__))


def train_network():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    train_datasets = ["0012", "0013", "0014","0015", "0016", "0017"]

    
    all_train_datasets = [CNNDataset(f"../data/DJI_{ds_num}", input_transform = transforms.ToTensor()) for ds_num in train_datasets]

    train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    
    

    cnn = SmallRegularizedCNN()
    cnn = cnn.cuda()
    optimizer = optim.AdamW(cnn.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
    patience = 10
    early_stopping = EarlyStopping(patience=patience, path= 'models/checkpoint.pt', verbose=True)
    
    num_epochs = 1000
    

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_train_accuracy = 0.0
        cnn.train()
        for data in tqdm(trainloader):
            img_feature, non_spatial_feature, labels = data
            img_feature = img_feature.cuda()
            non_spatial_feature = non_spatial_feature.cuda()
            labels = labels.cuda()
            cnn.forward(img_feature, non_spatial_feature)

            optimizer.zero_grad()

            preds = cnn(img_feature, non_spatial_feature)
            labels = labels.unsqueeze(1)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(trainloader)
            predictions = (preds > 0.5).float()
            correct = (predictions == labels).float().sum() / labels.shape[0]
            running_train_accuracy += correct / len(trainloader)
        
        top1, _, _ = get_predictions(cnn, "../data/DJI_0021")
        
        # We subtract 1 because early stopping is based on validation loss decreasing.
        early_stopping(1 - top1, cnn)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # print statistics
        
        print('[%d] loss: %.3f' % (epoch + 1, running_loss ))
        print('[%d] train accuracy: %.3f' % (epoch + 1, running_train_accuracy ))
        #print('[%d] validation accuracy: %.3f' % (epoch + 1, running_val_accuracy ))

    print('Finished Training')
    if not os.path.exists(_CURRENT + '/models'):
        os.mkdir(_CURRENT + '/models')

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    PATH = _CURRENT + '/models/smallRegularizedCNN_L%.3f_%s.pth' % (running_loss, timestamp)
    cnn.load_state_dict(torch.load(early_stopping.path))
    torch.save(cnn.state_dict(), PATH)
    

if __name__ == '__main__':
    train_network()
    
    