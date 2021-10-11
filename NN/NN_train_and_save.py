#!/usr/bin/evn python3
"""
This script performs data set loading and subsequent training of the model. Finally, it saves the output of the model:
- trained model in the .pt format
- loss value: plot and .txt file
Created by: Vice, 11.10.2021
"""
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import gc

# Append the required sys.path for accessing utilities and save the data directory
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.getcwd() + "/models"
os.chdir("..")
data_dir = os.getcwd() + "/data/data_NN"
sys.path.append(data_dir)
sys.path.append(models_dir)
os.chdir(curr_dir)

# Input the folder where a trained model should be saved and it's name - N.B. add the folder to .gitignore!
output_models_folder = curr_dir + "/output/trained_models"

from NN_params import *

# Load the dataset
from torch.utils.data import Dataset, DataLoader
from NN_dataset import bunniesDataset

train_split_size = 1 
test_split_size = 1

# Load Data
dataset = bunniesDataset(
    csv_file = data_dir + "/" + "annotations.csv",
    root_dir = data_dir,
)

train_set, test_set = torch.utils.data.random_split(dataset, [train_split_size, test_split_size])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

# Piece of code to check what is inside the loaded data - adjust to the shape of the simulation
test_iterator = iter(train_loader)
first = next(test_iterator)
print(first)

"""
# Define the models name. Naming convention based on the architecture, hyperparameters and dataset
model_name = "Selene_2ConvPool_MO_z_500_e_50_lr_em4_lRelu_0p1"

print("------ Start training Selene ------ \n")
print("Currently training: ", model_name)
# Ensure that the nondeterministic operations always have same starting point - increases the reproducibility of the code
torch.manual_seed(1)
np.random.seed(1) 

gc.collect() # Release the memmory

model.train()

time_start = time.time()

test_var = True
losses_all = []

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, x in enumerate(train_loader):
        # x = x.view(batch_size, x_dim) # This flattens the input tensor, can be done with transforms.ToTensor()
        x = x.to(DEVICE) # Save the batch to the device (CPU or GPU)

        optimizer.zero_grad() # Remove the gradients from the previous iteration
        
        # Feed the batch into VAE, compute the loss
        x_hat, mean, log_var = model(x.float())       
        loss = loss_function(x.float(), x_hat, mean, log_var)
        
        # Add the loss to the cumulative loss
        overall_loss += loss.item()
        
        # Backpropagate the loss and perform the optimizaiton with such gradients
        loss.backward()
        optimizer.step()
    
    loss_in_epoch = overall_loss / (batch_idx*batch_size)
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss_in_epoch)
    losses_all.append(loss_in_epoch)

time_end = time.time()    
print("------ Training finished!!! ------ \n")
print("Total training time: {0} min.".format( int((time_end - time_start) /60) ))

# Save the model
PATH_model = output_models_folder + "/" + model_name + ".pt"
PATH_state_dict = output_models_folder + "/" + model_name + "_state_dict"
torch.save(model, PATH_model)
torch.save(model.state_dict(), PATH_state_dict)

# Create the loss value output - relevant for model optimization
txt_file_loss_output = open(output_models_folder + "/" + model_name + "_loss.txt", "w")
for element in losses_all:
    txt_file_loss_output.write(str(element) + "\n")
txt_file_loss_output.close()

# Plot the loss
plt.figure()
plt.title(model_name + " " + "loss values")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses_all)
plt.show()

print("Model, state dictionary and loss function values saved at: ", output_models_folder + "/" + model_name)
"""