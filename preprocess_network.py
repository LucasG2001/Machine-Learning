import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, datasets
from torchvision.transforms import transforms
from PIL import Image
from triplet_loss_network import TripletNet

# get filename as string from image number (integer)
def getfilename(number):
    a = str(number)

    b = '0' * (5 - len(a))

    return b + a

val_transforms = transforms.Compose([
        transforms.Resize((300,1000)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#pre_model = model.resnet18(pretrained=True) Nutze das hier wenn das andere zu langsam ist
model = TripletNet()
model_loaded = model.to(device)  # load model
checkpoint = torch.load('model_epoch_7.pt')  # adjust manually for best epoch
model_loaded.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters
model_loaded.eval()

data_dir = "./TripletData_Train"

"""
image_datasets = {'A': datasets.ImageFolder(os.path.join(data_dir,'A_data'), val_transforms),
                  'B': datasets.ImageFolder(os.path.join(data_dir,'B_data'), val_transforms),
                  'C': datasets.ImageFolder(os.path.join(data_dir,'C_data'), val_transforms)
                  }
"""
total_dataset = datasets.ImageFolder("./Food/images/", val_transforms)
complete_dataloader = torch.utils.data.DataLoader(total_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=8, pin_memory=True,) for x in ['A', 'B', 'C']}

n = len(total_dataset)
A = np.zeros((n,1000))
B = np.zeros((n,1000))
C = np.zeros((n,1000))

image_features = np.zeros((n,50))
"""
for i, data in enumerate(complete_dataloader):
    input, labels = data
    input = input.to(device)
    embedding1, embedding2, embedding3 = model_loaded(input, input, input)
    embedding1 = embedding1.detach().cpu().numpy()
    image_features[i, :] = embedding1
    if i % 1000 == 0:
        print(image_features[i, :].shape)

image_features_df = pd.DataFrame(image_features)
image_features_df.to_csv('./preprocessed_data/TrianedNet50.csv')
"""

for i, data in enumerate(complete_dataloader):
    inputs, labels = data

    inputsA = inputs[0].to(device)
    inputsB = inputs[0].to(device)
    inputsC = inputs[0].to(device)

    pos, pos2, pos3 = model_loaded(inputsA, inputsB, inputsC)

    pos = pos.detach().cpu().numpy()

    image_features[i, :] = pos

    print(i)

image_features_df = pd.DataFrame(image_features)
image_features_df.to_csv('./preprocessed_data/TrainedNet50.csv')
"""
image_dir = "./Food/images/image/"
image_features = np.zeros((n,1000))
with torch.no_grad():
    for i in range(n):
        path = os.path.join(image_dir, getfilename(i)) + ".jpg"
        img = Image.open(path)
        img = image_transform(img)

        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        image_features[i,:] = (pre_model(img)).detach().cpu().numpy()
        if i % 1000 == 0:
            print(image_features[i,:].shape)

image_features_df = pd.DataFrame(image_features)
#image_features_df.to_csv('./preprocessed_data/BigNet.csv')

image_features_df.to_csv('./preprocessed_data/TrianedNet50.csv')



A_df = pd.DataFrame(A)
B_df = pd.DataFrame(B)
C_df = pd.DataFrame(C)


A_df.to_csv('./preprocessed_data/A.csv')
B_df.to_csv('./preprocessed_data/B.csv')
C_df.to_csv('./preprocessed_data/C.csv')
"""



