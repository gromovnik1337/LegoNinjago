#!/usr/bin/evn python3
"""
This script loads a pretrained NN model and .
Created by: Vice, 16.10.2021
"""
import torch
import torch.nn as nn
import numpy as np
import gc
from NN_dataset import test_loader
import math
from NN_params import *

# Append the required sys.path for accessing utilities and save the data directory
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.getcwd() + "/models/Bertybob"
os.chdir("..")
sys.path.append(models_dir)
os.chdir(curr_dir)

# Input the model name and the desired number of reconstructions
model_name = "Bertybob" # Same as in NN_train_and_save.py script
num_of_reconstructions = 10

model_path = models_dir + "/" + model_name + ".pt"
state_dict_path = model_path[:-3] + "_state_dict"

# Create the output folder
output_folder = curr_dir + "/output" + "/" + model_name
if not os.path.exists(output_folder):
    os.mkdir(output_folder) 

# Load the model and the parameters
from NN_params import *
from NN_model import *

gc.collect() # Release the memmory

model = torch.load(model_path)
model.load_state_dict(torch.load(state_dict_path))
model.eval()

print("------ Making inference ------ \n")
print("Model name: ", model_name)

with torch.no_grad():
    for y in test_loader:
        test = model(y)
        print(test)

print("Output saved at: ", output_folder)        
