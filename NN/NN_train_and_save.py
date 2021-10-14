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
from pathlib import Path
from torch import nn, optim
from NN_dataset import bunniesDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class BunnyRegressorNetwork(nn.Module):
    def __init__(self, in_channels, first_hidden, second_hidden, out_channels):
        super(BunnyRegressorNetwork, self).__init__()
        self.input_l    = nn.Linear(in_channels, first_hidden)
        self.hidden_l   = nn.Linear(first_hidden, second_hidden)
        self.output_l   = nn.Linear(second_hidden, out_channels)

    def forward(self, x):
        x   = F.relu(self.input_l(x))
        x   = F.relu(self.hidden_l(x))
        x   = self.output_l(x)

        return x

def LoadDataset(dataset_dir):
    dataset = bunniesDataset(
            csv_file = dataset_dir + "/" + "ErrorDataframe.csv",
            root_dir = dataset_dir,
            )

    return dataset

def SplitDataset(dataset, ratio):
    train_split_size = math.floor(ratio * len(dataset))
    test_split_size = len(dataset) - train_split_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_split_size, test_split_size])
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

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
    test_var = True
    losses_all = []

    optimizer   = optim.Adam(network.parameters(), lr = 0.005)
    criterion   = nn.MSELoss()

    print("------ Bertybob is now learning ------ \n")
    time_start = time.time()

    for epoch in range(epochs):
        running_loss    =   0
        iter_counter    =   0

        for batch_idx, (input, label) in enumerate(dataloader):
            input   = input.to(device) # Save the batch to the device (CPU or GPU)
            optimizer.zero_grad() # Remove the gradients from the previous iteration

            # Feed the batch into VAE, compute the loss
            output  = network(input)
            loss    = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_counter += 1

            '''
            if iter_counter % 4 == 3:
                print("Epoch: {:3d} | Ieration : {:3d} | Loss : {:3.3f}".format(epoch + 1, iter_counter + 1, running_loss))
                running_loss    =   0
            '''

            # Backpropagate the loss and perform the optimizaiton with such gradients

        loss_in_epoch = running_loss / (batch_idx*batch_size)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss_in_epoch)
        losses_all.append(loss_in_epoch)

    time_end = time.time()
    print("\n------ Training finished!!! ------ \n")
    print("Total training time: {0} min.".format( int((time_end - time_start) /60) ))

    return network, losses_all

def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Append the required sys.path for accessing utilities and save the data directory
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.getcwd() + "/models"
    os.chdir("..")
    data_dir = os.getcwd() + "/data/data_NN"
    sys.path.append(data_dir)
    sys.path.append(models_dir)
    os.chdir(curr_dir)

    output_models_folder    = curr_dir + "/output/trained_models" # Input the folder where a trained model should be saved and it's name - N.B. add the folder to .gitignore!
    model_name              = "Bertybob" # Define the models name. Naming convention based on the architecture, hyperparameters and dataset

    regression_network      = BunnyRegressorNetwork(input_dim, first_hidden, second_hidden, output_dim)
    dataset                 = LoadDataset(data_dir)
    trainloader, testloader = SplitDataset(dataset, 0.75)

    regression_network, loss_data = train(network = regression_network, dataloader = trainloader, epochs = 1000, device = 'cpu')
    SaveModel(models_dir, model_name, regression_network, loss_data)
    PlotLosses(model_name, loss_data)

if __name__ == '__main__':
    main(seed = 543001)
