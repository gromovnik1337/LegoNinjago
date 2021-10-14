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

    '''
    each of the above ojbects has the folllowing structure:

                    |-til (2D-np.array)
    plane_dataframe-|-hor (2D-np.array)
                    |-ver (2D-np.array)

    '''

    error_df = pd.DataFrame(columns = ["Bunny ID", "Plane", "X Error Mean", "Y Error Mean", "Z Error Mean", "X Error Std", "Y Error Std", "Z Error Std"])

    plot_active = False

    df_idx = 1
    for idx, bunny in enumerate([bunny_1_points, bunny_2_points, bunny_3_points]):

        if plot_active == True:
            fig = plt.figure(figsize = (7, 7))
            fig.suptitle("bunny_{}".format(idx + 1))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.set_title("Horizontal")
            ax1.scatter(x = bunny.hor[1, :], y = bunny.hor[0, :], color = 'blue')
            ax1.scatter(x = cad_points.hor[1, :], y = cad_points.hor[0, :], color = 'orange')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_title("Vertical")
            ax2.scatter(x = bunny.ver[1, :], y = bunny.ver[2, :], color = 'blue')
            ax2.scatter(x = cad_points.ver[1, :], y = cad_points.ver[2, :], color = 'orange')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_title("Tilt")
            ax3.scatter(x = bunny.til[1, :], y = bunny.til[2, :], color = 'blue')
            ax3.scatter(x = cad_points.til[1, :], y = cad_points.til[2, :], color = 'orange')
            plt.show()

        for plane in bunny.__dict__:
            cad_plane = cad_points.__getattribute__(plane)
            scan_plane = bunny.__getattribute__(plane)

            delta = cad_plane - scan_plane
            delta = np.round(delta, 3)

            delta_abs = np.array(list(map(abs, delta)))
            delta_mean = np.round(np.mean(delta_abs, axis = 1), 3)
            delta_std = np.round(np.std(delta_abs, axis = 1), 3)

            setattr(bunny, str(plane), delta_mean)

            error_entry = ["bunny_{}".format(idx + 1), str(plane), delta_mean[0], delta_mean[1], delta_mean[2], delta_std[0], delta_std[1], delta_std[2]]
            error_df.loc[df_idx] = error_entry
            df_idx += 1

        if plot_active == True:
            fig = plt.figure(figsize = (7, 7))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.set_title("Horizontal")
            ax1.scatter(x = bunny.hor[1], y = bunny.hor[0], color = 'red')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_title("Vertical")
            ax2.scatter(x = bunny.ver[1], y = bunny.ver[2], color = 'red')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_title("Tilt")
            ax3.scatter(x = bunny.til[1], y = bunny.til[2], color = 'red')
            plt.show()

    print(error_df)
    error_df.to_csv(data_dir + "/ErrorDataframe.csv", index = False)

    # Yet to decide metric to extract from cloud sphere datapoints...
