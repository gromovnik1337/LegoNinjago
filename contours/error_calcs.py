import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from cont_utils_plot_spheres import *
from cont_utlis_icp import LoadPointcloud, LoadPointcloudList
import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
sys.path.append(data_dir)
os.chdir(curr_dir)

if __name__ == '__main__':

    PlaneData = pd.read_csv(data_dir + "/" + "MeasurementDataset.csv").to_numpy()

    [cad_bunny_dps, bunny_1_dps, bunny_2_dps, bunny_3_dps, bunny_4_dps] = \
        LoadPointcloudList(data_dir, ["cad_bunny_pcd", "bunny_1_pcd", "bunny_2_pcd", "bunny_3_pcd", "bunny_4_pcd"])

    cad_planes      =   DirectToXYZObject(PlaneData, 'nominal')
    bunny_1_planes  =   DirectToXYZObject(PlaneData, 'bunny_1')
    bunny_2_planes  =   DirectToXYZObject(PlaneData, 'bunny_2')
    bunny_3_planes  =   DirectToXYZObject(PlaneData, 'bunny_3')
    bunny_4_planes  =   DirectToXYZObject(PlaneData, 'bunny_4')
    print('\n')

    print(cad_planes.ver.y)

    '''
    each of the above ojbects has the folllowing structure:

                          |-x (1D-np.array)
                    |-til-|-y (1D-np.array)
                    |     |-z (1D-np.array)
                    |
                    |     |-x (1D-np.array)
    plane_dataframe-|-hor-|-y (1D-np.array)
                    |     |-z (1D-np.array)
                    |
                    |     |-x (1D-np.array)
                    |-ver-|-y (1D-np.array)
                          |-z (1D-np.array)
    '''

    # Yet to decide metric to extract from cloud sphere datapoints...
