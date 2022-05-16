import torch
from custom_dataset import TripletDataset
from evaluation import accuracy, evaluate
from triplet_loss_network import TripletNet
from torchvision import models
import torch.optim as optim
import numpy as np
import time


def train_network_complete(model, train_dataloader, val_dataloader, device, number_epochs=30):
    save_epoch = 2
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    #optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay=0.1)
    criterion = torch.nn.TripletMarginLoss(margin=0.1, p=2)
    train_acc_arr = np.zeros(number_epochs)
    val_acc_arr = np.zeros(number_epochs)

    for epoch in range(number_epochs):
        # reset statistics trackers
        train_loss_cum = 0.0
        train_acc_cum = 0.0
        num_samples_epoch = 0
        t = time.time()

        for batch in train_dataloader:
            # extract images from train loader
            anchor, positive, negative = batch
            anchor = anchor.float().to(device)
            positive = positive.float().to(device)
            negative = negative.float().to(device)

            # zero grads and put model into train mode
            optim.zero_grad(set_to_none= True)
            model.train()

            ## forward pass
            out1, out2, out3 = model(anchor, positive, negative)

            ##compute loss
            loss = criterion(out1, out2, out3)

            ## backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = len(batch[0])
            num_samples_epoch += num_samples_batch

            train_loss_cum += loss * num_samples_batch

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_train_acc = evaluate(model, train_dataloader)
        avg_val_acc = evaluate(model, val_dataloader)
        epoch_duration = time.time() - t

        train_acc_arr[epoch] = avg_train_acc
        val_acc_arr[epoch] = avg_val_acc

        # print some infos
        print(
            f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | Train accuracy: {avg_train_acc:.4f} | Validation accuracy: {avg_val_acc:.4f} |'
            f' Duration {epoch_duration:.2f} sec')

        # save checkpoint of model
        if epoch % save_epoch == 0 and epoch > 0:
            save_path = f'model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict()},
                       save_path)
            print(f'Saved model checkpoint to {save_path}')

    save_path = f'model_final.pt'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()},
               save_path)
    print(f'Saved final model to {save_path}')

    return number_epochs, train_acc_arr, val_acc_arr