import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from custom_dataset import TripletDataset
from train import train_network_complete
from triplet_loss_network import TripletNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_triplets = pd.read_csv('./train_triplets.txt', header=None, names=["A", "B", "C"], delimiter=" ", dtype=str)
test_triplets = pd.read_csv('./test_triplets.txt', header=None, names=["A", "B", "C"], delimiter=" ", dtype=str)
triplets = {'train': train_triplets, 'test': test_triplets}

image_features = pd.read_csv('./preprocessed_data/BigNet.csv',dtype=float).to_numpy()
image_features = np.delete(image_features,0,1)
#image_features = np.random((10000, 1000))

datasets = {x: TripletDataset(triplets[x], image_features) for x in ['train', 'test']}  # size about 59500
print("Train Dataset Size: ", len(datasets['train']), " Test Dataset Size: ", len(datasets['test']))
test_dataloader = DataLoader(datasets['test'], shuffle=False, batch_size=32)
# start predictions
model = TripletNet()
model_loaded = model.to(device)  # load model
checkpoint = torch.load('model_final.pt')  # adjust manually for best epoch
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

np.savetxt('predictions_margin01_075split_b512_e30.txt',predictions.to('cpu'), newline='\n',fmt="%d")
print(len(predictions))