import torch.nn as nn
import torch


class TripletNet(torch.nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(1000, 1500),
                                nn.BatchNorm1d(1500),
                                nn.LeakyReLU(),
                                nn.Linear(1500,750),
                                nn.LeakyReLU(),
                                nn.Linear(750, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 250),
                                nn.LeakyReLU(),
                                nn.Linear(250, 50)
                                )

    def forward(self, a, p, n):
        pout = self.fc(p)
        nout = self.fc(n)
        aout = self.fc(a)

        return aout, pout, nout
