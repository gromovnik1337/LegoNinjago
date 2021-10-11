import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from NN_params import *

# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data/"
sys.path.append(data_dir)
os.chdir(curr_dir)

# Load in dataset of simulation bunnies and measurements
database = pd.read_csv(data_dir + 'DummyDataset.csv')

# Removing Bunny_4 as error measurements are incorrect
database.drop('bunny_4', axis = 1, inplace = True)

# Crop to build parameter inputs and convert to numpy array for model compatability
data_X = database.truncate(before = 0, after = 2)
data_X_cropped = data_X.drop(["Build Parameters", "Unit"], axis = 1)
data_X_np = data_X_cropped.to_numpy()

# Crop to measured error ouputs/labels and convert to numpy array for model compatability
data_Y = database.truncate(before = 4, after = database.shape[0] - 2)
data_Y.rename(columns = {"Build Parameters" : "Location", "Unit" : "Axis"}, inplace = True)
data_Y_cropped = data_Y.drop(["Location", "Axis"], axis = 1)
data_Y_np = data_Y_cropped.to_numpy()

# Print input targets for model
print("Input targets:\n{}\n".format(data_X_np))

# Print input labels for model
print("Input labels:\n{}\n".format(data_Y_np))

cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
torch.cuda.empty_cache() # Clean the memory

nn_model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), # M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, M), # H hidden units to M output neurons
                    torch.nn.Sigmoid() # final tranfer function
                    )

loss_fn = torch.nn.MSELoss()
