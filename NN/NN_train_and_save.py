#!/usr/bin/evn python3
"""
This script performs data set loading and subsequent training of the VAE CNN model. Finally, it saves the output of the model:
- trained model in the .pt format
- loss value: plot and .txt file
Created by: Vice, 17.07.2021
Updated by: Vice, 17.08.2021
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
utils_dir = os.getcwd() + "/utils"
data_dir = os.getcwd() + "/data"
sys.path.append(utils_dir)
sys.path.append(data_dir)
sys.path.append(models_dir)
os.chdir(curr_dir)

# Input the folder where a trained model should be saved and it's name - N.B. add the folder to .gitignore!
output_models_folder = curr_dir + "/output/trained_models"

# Import the model and the parameters
#from ANN_params_3Conv import *
#from ANN_Selene_3Conv import * #TODO Change this, this is not a good practice!

from ANN_params_2ConvPool import *
from ANN_Selene_2ConvPool import * #TODO Change this, this is not a good practice!

# Load the dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils_ANN_data_prep import stepHeightPointCloudDataset

# Input the folders with the downsampled and normalized data in the fixed grid 3D image format, containing both top and the bottom surfaces
sample_folder = data_dir + "/measuring_object_5x/dwnsmpl_th_clean_norm_merged_MO"
#sample_folder = data_dir + "/calibration_measurements_first/dwnsampled_th_clean_norm_merged_cali"

train_split_size = 41 
test_split_size = 9

# Load Data
dataset = stepHeightPointCloudDataset(
    csv_file = sample_folder + "/" + "annotations.csv",
    root_dir = sample_folder,
    #transform = transforms.Compose([
    #transforms.ToTensor(),
#])
)

train_set, test_set = torch.utils.data.random_split(dataset, [train_split_size, test_split_size])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

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