#!/usr/bin/evn python3
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import sys
from NN_params import *
import math
from torch.utils.data import Dataset, DataLoader

# Append the required sys.path for accessing utilities and save the data directory
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data/data_NN"
sys.path.append(data_dir)
os.chdir(curr_dir)

class bunniesDataset_training(Dataset):
    """A class that makes Pytorch data set.
    Sources:
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset/custom_dataset.py
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1

    Args:
        Dataset (Class): An abstract class representing a dataset, inherited. All the other pytorch datasets should subclass it.
        It contains structures used below such as transformations.
        csv_file (string): Absolute path to the csv file containing the data set annotations = names of the data set entries.
        root_dir (string): Absolute path to the directory containing all the data sets.
    """

    def __init__(self, csv_file, root_dir):
        self.root_dir   = root_dir
        self.dataframe  = pd.read_csv(os.path.join(self.root_dir, csv_file))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        X = torch.tensor(self.dataframe.iloc[index, 1:10], dtype = torch.float32) # Measurement results
        y = torch.tensor(self.dataframe.iloc[index, 10:13], dtype = torch.float32) # Printing parameters

        return X, y

class bunniesDataset_test(Dataset):
    """A class that makes Pytorch data set.
    Sources:
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset/custom_dataset.py
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1

    Args:
        Dataset (Class): An abstract class representing a dataset, inherited. All the other pytorch datasets should subclass it.
        It contains structures used below such as transformations.
        csv_file (string): Absolute path to the csv file containing the data set annotations = names of the data set entries.
        root_dir (string): Absolute path to the directory containing all the data sets.
    """

    def __init__(self, csv_file, root_dir):
        self.root_dir   = root_dir
        self.dataframe  = pd.read_csv(os.path.join(self.root_dir, csv_file))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        X = torch.tensor(self.dataframe.iloc[index, 1:10], dtype = torch.float32) # Measurement results

        return X


dataset_training = bunniesDataset_training(
        csv_file = data_dir + "/" + "Sim_ALL_features.csv",
        root_dir = data_dir,
            )

dataset_testing = bunniesDataset_test(
        csv_file = data_dir + "/" + "Sim_test_features.csv",
        root_dir = data_dir,
            )

train_split_size = math.floor(ratio * len(dataset_training))
test_split_size = math.floor(ratio * len(dataset_testing))

train_loader = DataLoader(dataset = dataset_training, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = dataset_testing, batch_size = batch_size, shuffle = False)

