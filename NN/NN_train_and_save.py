#!/usr/bin/evn python3
"""
This script performs data set loading and subsequent training of the model. Finally, it saves the output of the model:
- trained model in the .pt format
- loss value: plot and .txt file
Created by: Vice, 11.10.2021
"""

import os
import gc
import sys
import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from NN_params import *
from NN_model import *
from pathlib import Path
from torch import nn, optim
from NN_dataset import train_loader
from torch.nn import functional as F

# Append the required sys.path for accessing utilities and save the data directory
curr_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.getcwd() + "/models"
os.chdir("..")
data_dir = os.getcwd() + "/data/data_NN"
sys.path.append(data_dir)
sys.path.append(models_dir)
os.chdir(curr_dir)


def SaveModel(output_dir, model_name, model, loss_data):
    model_dir = output_dir + '/' + model_name + '/'
    Path(model_dir).mkdir(parents = True, exist_ok = True)
    Path(model_dir).mkdir(parents = True, exist_ok = True)
    torch.save(model, model_dir + model_name + ".pt")
    torch.save(model.state_dict(), model_dir + model_name + "_state_dict")

    # Create the loss value output - relevant for model optimization
    txt_file_loss_output = open(model_dir + "/" + model_name + "_loss.txt", "w")
    for element in loss_data:
        txt_file_loss_output.write(str(element) + "\n")
    txt_file_loss_output.close()

    print("Model, state dictionary and loss function values saved at: ", model_dir)

def PlotLosses(model_name, loss_data):
        plt.figure(figsize = (12, 7))
        plt.suptitle(model_name + " loss values")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(loss_data)
        plt.show()

def train(network, dataloader, epochs, device):
    gc.collect()
    network.train()

    losses_all = []

    print("------ Bertybob is now learning ------ \n")
    time_start = time.time()

    for epoch in range(epochs):
        running_loss    =   0
        iter_counter    =   0

        for batch_idx, (input, label) in enumerate(dataloader):
            input   = input.to(device) # Save the batch to the device (CPU or GPU)
            optimizer.zero_grad() # Remove the gradients from the previous iteration

            # Feed the batch into the model, compute the loss
            output  = network(input) # Perform the forward pass
            loss    = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_counter += 1

        # Backpropagate the loss and perform the optimizaiton with such gradients
        loss_in_epoch = running_loss / (batch_idx*batch_size)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss_in_epoch)
        losses_all.append(loss_in_epoch)

    time_end = time.time()
    print("\n------ Training finished!!! ------ \n")
    print("Total training time: {0} min.".format( int((time_end - time_start) /60) ))

    return network, losses_all

def main(model, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    #output_models_folder    = curr_dir + "/output/trained_models" # Input the folder where a trained model should be saved and it's name - N.B. add the folder to .gitignore!
    model_name              = "Bertybob" # Define the models name. Naming convention based on the architecture, hyperparameters and dataset

    trainloader              = train_loader

    #test_iterator = iter(trainloader)
    #first = next(test_iterator)
    #print(first)

    regression_network, loss_data = train(network = model, dataloader = trainloader, epochs = epochs, device = 'cpu')

    SaveModel(models_dir, model_name, regression_network, loss_data)
    PlotLosses(model_name, loss_data)

if __name__ == '__main__':
    main(model = regression_network, seed = 543001)
