import torch
import torch.nn as nn
from NN_params import *
from torch import nn, optim

cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
torch.cuda.empty_cache() # Clean the memory

class BunnyRegressorNetwork(nn.Module):
    def __init__(self, in_channels, first_hidden, second_hidden, out_channels):
        super(BunnyRegressorNetwork, self).__init__()
        self.input_l    = nn.Linear(in_channels, first_hidden)
        self.hidden_l_1   = nn.Linear(first_hidden, second_hidden)
        self.hidden_l_2   = nn.Linear(second_hidden, second_hidden)
        self.output_l   = nn.Linear(second_hidden, out_channels)
        #self.transferFunction = nn.LeakyReLU(lRelu_neg_slope)
        self.transferFunction = nn.Tanh()
        #self.transferFunction_last = nn.Sigmoid()

    def forward(self, X):
        h_1   = self.transferFunction(self.input_l(X))
        h_2   = self.transferFunction(self.hidden_l_1(h_1))
        h_3   = self.transferFunction(self.hidden_l_1(h_2))
        y   = self.output_l(h_3)

        return y

regression_network      = BunnyRegressorNetwork(input_dim, first_hidden, second_hidden, output_dim).to(DEVICE) # Model

optimizer   = optim.Adam(regression_network.parameters(), lr = lr)
criterion   = nn.MSELoss()



