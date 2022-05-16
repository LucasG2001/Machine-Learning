import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pdist = torch.nn.PairwiseDistance(p=2)
# compute prediction accuracy given the NN outputs
def accuracy(anchor, pos, neg):
    d1 = pdist(anchor, pos)
    d2 = pdist(anchor, neg)

    prediction = (d1 < d2)
    score = torch.count_nonzero(prediction)

    assert score <= len(prediction)

    return score


# evaluate trained model on validation data
def evaluate(ev_model: torch.nn.Module, ev_dataloader) -> torch.Tensor:
    # goes through the test dataset and computes the test accuracy
    ev_model.eval()  # bring the model into eval mode
    with torch.no_grad():
        acc_cum = 0.0
        num_eval_samples = 0
        for batch in ev_dataloader:
            im1, im2, im3 = batch
            im1 = im1.float().to(device)
            im2 = im2.float().to(device)
            im3 = im3.float().to(device)

            num_samples_batch = len(batch[0])
            num_eval_samples += num_samples_batch
            out1, out2, out3 = ev_model(im1, im2, im3)
            acc_cum += accuracy(out1, out2, out3)
        avg_acc = acc_cum / num_eval_samples
        assert 0 <= avg_acc <= 1
        return avg_acc


def plot(number_epochs, train_acc, val_acc):
    x_axis = np.arange(number_epochs)
    plt.plot(x_axis, train_acc, label='Training accuracy')
    plt.plot(x_axis, val_acc, label='Validation accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Accuracy_Plot.png')