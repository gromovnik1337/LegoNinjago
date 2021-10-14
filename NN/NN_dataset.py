#!/usr/bin/evn python3
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from numpy import genfromtxt

class bunniesDataset(Dataset):
    """A class that makes Pytorch data set out of the point clouds saved as .csv (txt) files. To load the dataset,
    one must provide a folder containing the data and annotations.csv file with the names of each cloud (IDs of the data set entries) and,
    if provided, labels that each of the entry has.
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
        X = torch.tensor(self.dataframe.iloc[index, 1:10], dtype = torch.float32)
        y = torch.tensor(self.dataframe.iloc[index, 10:13], dtype = torch.float32)

        return X, y
