from network import SimpleCNN
from utils import CNNDataset
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
from network import SimpleCNN
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
    num_epochs=25
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
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
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rates[fold])
        
        loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
        
        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                
                # Get inputs
                inputs, targets = data
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = network(inputs)
                
                # Compute loss
                loss = loss_fn(outputs, targets)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                
                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                    current_loss = 0.0
                    
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        
        # Saving the model
        save_path = f'models//model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
        
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
    
    