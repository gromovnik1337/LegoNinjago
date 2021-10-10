import torch
import torch.nn as nn
from NN_params import *

# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
sys.path.append(data_dir)
os.chdir(curr_dir)

cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
torch.cuda.empty_cache() # Clean the memory

model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), # M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, M), # H hidden units to M output neurons
                    torch.nn.Sigmoid() # final tranfer function
                    )
                    
loss_fn = torch.nn.MSELoss()


