import pandas as pd
import numpy as np

# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
sys.path.append(data_dir)
os.chdir(curr_dir)

dataframe = pd.read_csv(data_dir + "/" + "MeasurementDataset.csv")
dataframe.dropna(inplace = True)

for count, string_name in enumerate(dataframe["Fingerprint Point"]):
    name_listed = string_name.split(' ')
    returned_name = name_listed[len(name_listed) - 1]
    print(returned_name)
    dataframe.at[count, "Fingerprint Point"] = returned_name
