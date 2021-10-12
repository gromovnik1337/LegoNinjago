#!/usr/bin/evn python3
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from cont_utils_plot_spheres import CorrectAxes, ConvertToXYZ, AffineFit, VerticalAxesSwitch, CreateContours, set_axes_equal
from cont_utlis_icp import LoadPointcloud, LoadPointcloudList

# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
sys.path.append(data_dir)
os.chdir(curr_dir)

if __name__ == '__main__':

    # Load the data and the clouds
    PlaneData = pd.read_csv(data_dir + "/" + "MeasurementDataset.csv").to_numpy()

    [cad_bunny_dps, bunny_1_dps, bunny_2_dps, bunny_3_dps, bunny_4_dps] = \
        LoadPointcloudList(data_dir, ["cad_bunny_pcd", "bunny_1_pcd", "bunny_2_pcd", "bunny_3_pcd", "bunny_4_pcd"])

    # Preliminary rotations (no longer need to be applied as global rotations and alignments have been saved to pointcloud data)
    cad_rotation_angle = math.pi / 2 * 0
    bunny_1_rotation_angle = math.pi / 2 * 2
    translation_distance_x = 14.897
    translation_distance_y = 8.3661
    translation_distance_z = 9.7559

    '''
    cad_bunny_dps = cad_bunny_dps.rotate(np.array([[math.cos(cad_rotation_angle), -1 * math.sin(cad_rotation_angle), 0],\
                                [math.sin(cad_rotation_angle), math.cos(cad_rotation_angle), 0],\
                                [0, 0, 1]]))

    bunny_1_dps = bunny_1_dps.rotate(np.array([[math.cos(bunny_1_rotation_angle), -1 * math.sin(bunny_1_rotation_angle), 0],\
                                [math.sin(bunny_1_rotation_angle), math.cos(bunny_1_rotation_angle), 0],\
                                [0, 0, 1]]))

    cad_bunny_dps = cad_bunny_dps.translate(np.array([1 * translation_distance_x, 1 * translation_distance_y, -1 * translation_distance_z]))

    '''

    cad_bunny_pts = np.asarray(cad_bunny_dps.points)
    cad_bunny_pts = CorrectAxes(cad_bunny_pts)

    bunny_1_pts = np.asarray(bunny_1_dps.points)
    bunny_1_pts = CorrectAxes(bunny_1_pts)

    bunny_2_pts = np.asarray(bunny_2_dps.points)
    bunny_2_pts = CorrectAxes(bunny_2_pts)

    bunny_3_pts = np.asarray(bunny_3_dps.points)
    bunny_3_pts = CorrectAxes(bunny_3_pts)

    bunny_4_pts = np.asarray(bunny_4_dps.points)
    bunny_4_pts = CorrectAxes(bunny_4_pts)

    # Terrible coding, I can only apologise...
    tilt_dps = ConvertToXYZ(PlaneData, 'nominal', 'tilt')
    tilt_dps = CorrectAxes(tilt_dps)
    hor_dps = ConvertToXYZ(PlaneData, 'nominal', 'horizontal')
    hor_dps = CorrectAxes(hor_dps)
    ver_dps = ConvertToXYZ(PlaneData, 'nominal', 'vertical')
    ver_dps = CorrectAxes(ver_dps)

    fig = plt.figure(figsize = (16, 16))

    tilt_cen, tilt_norms = AffineFit(tilt_dps)

    tilt_x, tilt_y = np.meshgrid(range(int(tilt_cen[0] - 15), int(tilt_cen[0] + 15)), range(int(tilt_cen[1] - 15), int(tilt_cen[1] + 15)))
    tilt_z = (np.ones(tilt_x.shape) * (tilt_cen[2]))
    tilt_z = tilt_z - ((tilt_x * tilt_norms[0]) / tilt_norms[2])
    tilt_z = tilt_z + ((tilt_cen[0] * tilt_norms[0]) / tilt_norms[2])

    hor_cen, _ = AffineFit(hor_dps)
    hor_x, hor_y = np.meshgrid(range(int(hor_cen[0] - 15), int(hor_cen[0] + 15)), range(int(hor_cen[1] - 15), int(hor_cen[1] + 15)))
    hor_z = np.ones(hor_x.shape) * hor_cen[2]

    ver_cen, _ = AffineFit(ver_dps)
    ver_y, ver_z = np.meshgrid(range(int(ver_cen[1] - 15), int(ver_cen[1] + 15)), range(int(ver_cen[2] - 15), int(ver_cen[2] + 15)))
    ver_x = np.ones(ver_y.shape) * ver_cen[0]

    ax = fig.add_subplot(projection = '3d')
    ax.scatter(xs = cad_bunny_pts[0, ::5], ys = cad_bunny_pts[1, ::5], zs = cad_bunny_pts[2, ::5], c = 'gray', alpha = 0.2, label = 'CAD Bunny', s = 1)
    ax.plot_surface(tilt_x, tilt_y, tilt_z, color = 'red', alpha = 0.1)
    ax.plot_surface(hor_x, hor_y, hor_z, color = 'blue', alpha = 0.1)
    ax.plot_surface(ver_x, ver_y, ver_z, color = 'green', alpha = 0.1)
    ax.scatter(xs = tilt_dps[0, :], ys = tilt_dps[1, :], zs = tilt_dps[2, :], c = 'red', label = '45$^\circ$ Fingerprints')
    ax.scatter(xs = hor_dps[0, :], ys = hor_dps[1, :], zs = hor_dps[2, :], c = 'blue', label = 'Horizontal Fingerprints')
    ax.scatter(xs = ver_dps[0, :], ys = ver_dps[1, :], zs = ver_dps[2, :], c = 'green', label = 'Vertical Fingerprints')
    set_axes_equal(ax)
    ax.legend()
    plt.show()

    CAD_tilt_contour = CreateContours(cad_bunny_pts, tilt_dps, 0.08, 'tilt')
    CAD_hor_contour = CreateContours(cad_bunny_pts, hor_dps, 0.08, 'horizontal')
    CAD_ver_contour = CreateContours(cad_bunny_pts, ver_dps, 0.08, 'vertical')

    bunny_1_tilt_contour = CreateContours(bunny_1_pts, tilt_dps, 0.08, 'tilt')
    bunny_1_hor_contour = CreateContours(bunny_1_pts, hor_dps, 0.08, 'horizontal')
    bunny_1_ver_contour = CreateContours(bunny_1_pts, ver_dps, 0.08, 'vertical')

    bunny_2_tilt_contour = CreateContours(bunny_2_pts, tilt_dps, 0.08, 'tilt')
    bunny_2_hor_contour = CreateContours(bunny_2_pts, hor_dps, 0.08, 'horizontal')
    bunny_2_ver_contour = CreateContours(bunny_2_pts, ver_dps, 0.08, 'vertical')

    bunny_3_tilt_contour = CreateContours(bunny_3_pts, tilt_dps, 0.08, 'tilt')
    bunny_3_hor_contour = CreateContours(bunny_3_pts, hor_dps, 0.08, 'horizontal')
    bunny_3_ver_contour = CreateContours(bunny_3_pts, ver_dps, 0.08, 'vertical')

    bunny_4_tilt_contour = CreateContours(bunny_4_pts, tilt_dps, 0.08, 'tilt')
    bunny_4_hor_contour = CreateContours(bunny_4_pts, hor_dps, 0.08, 'horizontal')
    bunny_4_ver_contour = CreateContours(bunny_4_pts, ver_dps, 0.08, 'vertical')

    fig = plt.figure(figsize = (14, 14))
    ax = fig.add_subplot()
    ax.scatter(x = CAD_tilt_contour[1, :], y = CAD_tilt_contour[2, :], color = 'blue', s = 2, label = 'CAD Bunny')
    ax.scatter(x = bunny_1_tilt_contour[1, :], y = bunny_1_tilt_contour[2, :], color = 'red', s = 2, label = 'Bunny 1')
    ax.scatter(x = bunny_2_tilt_contour[1, :], y = bunny_2_tilt_contour[2, :], color = 'blue', s = 2, label = 'Bunny 2')
    ax.scatter(x = bunny_3_tilt_contour[1, :], y = bunny_3_tilt_contour[2, :], color = 'green', s = 2, label = 'Bunny 3')
    ax.scatter(x = bunny_4_tilt_contour[1, :], y = bunny_4_tilt_contour[2, :], color = 'orange', s = 2, label = 'Bunny 4')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    fig = plt.figure(figsize = (14, 14))
    ax = fig.add_subplot()
    ax.scatter(x = CAD_hor_contour[1, :], y = CAD_hor_contour[0, :],  color = 'blue', s = 2, label = 'CAD Bunny')
    ax.scatter(x = bunny_1_hor_contour[1, :], y = bunny_1_hor_contour[0, :], color = 'red', s = 2, label = 'Bunny 1')
    ax.scatter(x = bunny_2_hor_contour[1, :], y = bunny_2_hor_contour[0, :], color = 'blue', s = 2, label = 'Bunny 2')
    ax.scatter(x = bunny_3_hor_contour[1, :], y = bunny_3_hor_contour[0, :], color = 'green', s = 2, label = 'Bunny 3')
    ax.scatter(x = bunny_4_hor_contour[1, :], y = bunny_4_hor_contour[0, :], color = 'orange', s = 2, label = 'Bunny 4')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

    fig = plt.figure(figsize = (14, 14))
    ax = fig.add_subplot()
    ax.scatter(x = CAD_ver_contour[1, :], y = CAD_ver_contour[2, :],  color = 'blue', s = 2, label = 'CAD Bunny')
    ax.scatter(x = bunny_1_ver_contour[1, :], y = bunny_1_ver_contour[2, :], color = 'red', s = 2, label = 'Bunny 1')
    ax.scatter(x = bunny_2_ver_contour[1, :], y = bunny_2_ver_contour[2, :], color = 'blue', s = 2, label = 'Bunny 2')
    ax.scatter(x = bunny_3_ver_contour[1, :], y = bunny_3_ver_contour[2, :], color = 'green', s = 2, label = 'Bunny 3')
    ax.scatter(x = bunny_4_ver_contour[1, :], y = bunny_4_ver_contour[2, :], color = 'orange', s = 2, label = 'Bunny 4')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

    '''
    bunny_4_dps = ConvertToXYZ(Til_PlaneData, 'bunny_4', 'horizontal')
    bunny_4_dps = CorrectAxes(bunny_4_dps)

    bunny_4_dps_conv = np.zeros(bunny_4_dps.shape)
    bunny_4_dps_conv[0, :], bunny_4_dps_conv[1, :], bunny_4_dps_conv[2, :] = bunny_4_dps[2, :], bunny_4_dps[0, :], bunny_4_dps[1, :]
    '''
