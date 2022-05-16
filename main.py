import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
import sys
import time
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import random
import matplotlib.pyplot as plt
from evaluation import accuracy, evaluate, plot
from custom_dataset import TripletDataset
from train import train_network_complete
from triplet_loss_network import TripletNet

rand_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300,300)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Fix Seed
torch.manual_seed(12)
np.random.seed(12)
random.seed(12)

train_size = 44600
val_size = 14875

train_triplets = pd.read_csv('./train_triplets.txt', header=None, names=["A", "B", "C"], delimiter=" ", dtype=str)
test_triplets = pd.read_csv('./test_triplets.txt', header=None, names=["A", "B", "C"], delimiter=" ", dtype=str)
triplets = {'train': train_triplets, 'test': test_triplets}

image_features = pd.read_csv('./preprocessed_data/BigNet.csv',dtype=float).to_numpy()
image_features = np.delete(image_features,0,1)
#image_features = np.random((10000, 1000))

datasets = {x: TripletDataset(triplets[x], image_features) for x in ['train', 'test']}
dataset_augmented = TripletDataset(triplets['train'], image_features, transform=rand_transform)# size about 59500
print("Train Dataset Size: ", len(datasets['train']), " Test Dataset Size: ", len(datasets['test']))

used_datapoints = list(range(0, train_size + val_size))  # Define used inicies of complete dataset
dataset_reduced = Subset(datasets['train'], used_datapoints)
#aug_reduced = Subset(dataset_augmented, used_datapoints)  # Define Subset for performance
[train_data, val_data] = random_split(dataset_reduced, [train_size, val_size])
#[aug_train, aug_val] = random_split(aug_reduced, [train_size, val_size])# Split into test and training data
#data_train = ConcatDataset([train_data, aug_train])
#data_val = ConcatDataset([val_data, aug_val])
# Define dataloaders for training and validation
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=512, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_data, shuffle=True, batch_size=512, num_workers=8, pin_memory=True)

#aug_train_loader = DataLoader(aug_train, shuffle=True, batch_size=1024, num_workers=8, pin_memory=True)
# Define dataloaders for testing set (note: Shuffle=False to preserve order)
test_dataloader = DataLoader(datasets['test'], shuffle=False, batch_size=256, num_workers=8, pin_memory=True)

model = TripletNet()
model = model.to(device)
print(model)

# go through complete training loop
n_epochs, train_acc, val_acc = train_network_complete(model, train_dataloader, val_dataloader, device, number_epochs=30)
print("plotting")
plot(n_epochs, train_acc, val_acc)

# start predictions
model_loaded = model.to(device)  # load model
checkpoint = torch.load('model_epoch_22.pt')  # adjust manually for best epoch
model_loaded.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters

# put model in evaluation mode
model_loaded.eval()

predictions = torch.empty(0, device=device)
pdist = torch.nn.PairwiseDistance(p=2)

with torch.no_grad():
    for j, (batch) in enumerate(test_dataloader):
        # load data from dataloader
        im1, im2, im3 = batch
        im1 = im1.float().to(device)
        im2 = im2.float().to(device)
        im3 = im3.float().to(device)

        # forward pass
        out1, out2, out3 = model_loaded(im1, im2, im3)

        # compute similarity scores for image paris A-B and A-C
        d1 = pdist(out1, out2)
        d2 = pdist(out1, out3)

        # piecewise comparison of similarity score of pairs A-B and A-C
        batch_out = d1 < d2

        predictions = torch.cat((predictions, batch_out), dim=0)

        # print progress of prediction process
        if j % 10 == 0:
            print(j)

print(torch.count_nonzero(predictions))

np.savetxt('predictions.txt',predictions.to('cpu'), newline='\n',fmt="%d")
print(len(predictions))