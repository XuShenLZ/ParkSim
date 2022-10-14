from parksim.intent_predict.cnnV2.network import SimpleCNN
from parksim.intent_predict.cnnV2.utils import CNNDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

import os
from datetime import datetime
import numpy as np
from parksim.intent_predict.cnnV2.network import SimpleCNN
from sklearn.model_selection import KFold


_CURRENT = os.path.abspath(os.path.dirname(__file__))

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    
    
    dataset = CNNDataset("data/DJI_0012", input_transform = transforms.ToTensor())
    #trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

    results = {}
    num_folds = 5
    num_epochs= 15
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    kfold = KFold(n_splits=num_folds, shuffle=True)

    for lr_idx, learning_rate in enumerate(learning_rates):
        print(f"STARTING CV FOR LEARNING RATE = {learning_rate}:")
        print("=================================\n\n")
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=64, sampler=train_subsampler, num_workers=10)
            testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=64, sampler=test_subsampler, num_workers=10)
            
            # Init the neural network
            network = SimpleCNN().cuda()
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
            
            loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
            
            # Run the training loop for defined number of epochs
            for epoch in range(0, num_epochs):

                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                    
                    img_feature, non_spatial_feature, labels = data
                    img_feature = img_feature.cuda()
                    non_spatial_feature = non_spatial_feature.cuda()
                    labels = labels.cuda().unsqueeze(1)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Perform forward pass
                    preds = network(img_feature, non_spatial_feature)
                    
                    # Compute loss
                    loss = loss_fn(preds, labels)
                    
                    # Perform backward pass
                    loss.backward()
                    
                    # Perform optimization
                    optimizer.step()
                    
                    # Print statistics
                    current_loss += loss.item()
                    if i % 50 == 49:
                        print('Loss after mini-batch %5d: %.3f' %
                            (i + 1, current_loss / 50))
                        current_loss = 0.0
                        
            # Process is complete.
            print('Training process has finished. Saving trained model.')

            # Print about testing
            print('Starting testing')
            
            # Saving the model
            save_path = f'models//model-fold-{fold}-lr-{lr_idx}.pth'
            torch.save(network.state_dict(), save_path)

            # Evaluationfor this fold
            correct, total = 0, 0
            running_accuracy = 0
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    img_feature, non_spatial_feature, labels = data
                    img_feature = img_feature.cuda()
                    non_spatial_feature = non_spatial_feature.cuda()
                    labels = labels.cuda().unsqueeze(1)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Perform forward pass
                    preds = network(img_feature, non_spatial_feature)
                    
                    # Compute loss
                    loss = loss_fn(preds, labels)

                    # Set total and correct
                    predictions = (preds > 0.5).float()
                    correct = (predictions == labels).float().sum() / labels.shape[0]
                    running_accuracy += correct / len(testloader)

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * running_accuracy))
            print('--------------------------------')
            results[fold] = 100.0 * running_accuracy
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(results.items())} %')
    

if __name__ == '__main__':
    main()
    
    