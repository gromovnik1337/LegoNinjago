#!/usr/bin/evn python3
from torch import nn, optim

input_dim       = 9 # Size of the one flattened data set input
first_hidden    = 20
second_hidden   = 20
output_dim      = 3

lr          = 1e-4
epochs      = 30
batch_size  = 2
