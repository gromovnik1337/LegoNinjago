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

    cad_points      =   DirectToXYZObject(PlaneData, 'nominal')
    bunny_1_points  =   DirectToXYZObject(PlaneData, 'bunny_1')
    bunny_2_points  =   DirectToXYZObject(PlaneData, 'bunny_2')
    bunny_3_points  =   DirectToXYZObject(PlaneData, 'bunny_3')
    bunny_4_points  =   DirectToXYZObject(PlaneData, 'bunny_4')

    print(bunny_3_points.til) # This does correlate to what is found in the MeasurementDataset.csv file for the X values in the vertical plane for bunny 4

    '''
    each of the above ojbects has the folllowing structure:

                    |-til (2D-np.array)
    plane_dataframe-|-hor (2D-np.array)
                    |-ver (2D-np.array)

    '''

    print(bunny_1_points.__dir__())

    for bunny in [bunny_1_points, bunny_2_points, bunny_3_points, bunny_4_points]:
        for plane in bunny.__dict__:
            cad_plane = cad_points.__getattribute__(plane)
            scan_plane = bunny.__getattribute__(plane)

            delta = cad_plane - scan_plane
            delta = np.round(delta, 3)

            setattr(bunny, str(plane), delta)

    print(bunny_3_points.til)

    # Yet to decide metric to extract from cloud sphere datapoints...
