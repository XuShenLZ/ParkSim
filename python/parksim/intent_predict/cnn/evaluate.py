import torch
from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.intent_predict.cnn.utils import CNNDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

def main():
    model = SmallRegularizedCNN()
    model_state = torch.load('models/SmallRegularizedIntent-4-10-22.pt')
    model.load_state_dict(model_state)
    model.eval().cuda()
    dataset = CNNDataset("../data/DJI_0022", input_transform = transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12)
    running_accuracy = 0
    for data in tqdm(dataloader):
        img_feature, non_spatial_feature, labels = data
        img_feature = img_feature.cuda()
        non_spatial_feature = non_spatial_feature.cuda()
        labels = labels.cuda()
        #model.forward(img_feature, non_spatial_feature)
        #inputs, labels = data[0].to(device), data[1].to(device)

        #optimizer.zero_grad()

        preds = model(img_feature, non_spatial_feature)
        labels = labels.unsqueeze(1)
        #loss = loss_fn(preds, labels)

        #loss.backward()
        #optimizer.step()

        #running_loss += loss.item() / len(trainloader)
        preds = torch.nn.functional.sigmoid(preds)
        predictions = (preds > 0.5).float()
        correct = (predictions == labels).float().sum() / labels.shape[0]
        running_accuracy += correct / len(dataloader)
    print(f"Accuracy: {running_accuracy}")

if __name__ == '__main__':
    main()