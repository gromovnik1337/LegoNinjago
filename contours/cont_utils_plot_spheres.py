#!/usr/bin/evn python3
import numpy as np
import matplotlib.pyplot as plt

def CorrectAxes(datapoint_set):
    X_axis = datapoint_set[:, 0]
    Y_axis = datapoint_set[:, 1]
    Z_axis = datapoint_set[:, 2]

    switched_dataset = np.array([X_axis, Y_axis, Z_axis])
    return switched_dataset

def ConvertToXYZ(dataframe, bunny, plane):
    bunny = list(bunny.lower())
    plane = plane.lower()

    if plane == 'tilt':
        selection_min = 27
        selection_max = 52

    elif plane == 'horizontal':
        selection_min = 54
        selection_max = 79

    elif plane == 'vertical':
        selection_min = 0
        selection_max = 27

    df_z = dataframe[selection_min : selection_max : 3]
    df_y = dataframe[selection_min + 1 : selection_max + 1 : 3]
    df_x = dataframe[selection_min + 2 : selection_max + 2 : 3]

    if bunny == list('nominal'):
        idx = 2
    elif '1' in bunny:
        idx = 3
    elif '2' in bunny:
        idx = 4
    elif '3' in bunny:
        idx = 5
    elif '4' in bunny:
        idx = 6
    else:
        raise Exception('No valid bunny selected...')

    bunny_datapoints = np.block([df_x[:, idx].reshape(-1, 1), df_y[:, idx].reshape(-1, 1), df_z[:, idx].reshape(-1, 1)])
    return bunny_datapoints

def DirectToXYZObject(dataframe, bunny):

    print("Creating dataframe object/class containing datapoints for each plane i.e. (.til, .hor, .ver)...")
    class point_coords:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class plane_dataframe:
        def __init__(self, til, hor, ver):
            self.til = til
            self.hor = hor
            self.ver = ver

    converted_til = ConvertToXYZ(dataframe, bunny, "tilt")
    converted_til = CorrectAxes(converted_til)
    converted_til = point_coords(converted_til[0, :], converted_til[1, :], converted_til[2, :])

    converted_hor = ConvertToXYZ(dataframe, bunny, "horizontal")
    converted_hor = CorrectAxes(converted_hor)
    converted_hor = point_coords(converted_hor[0, :], converted_hor[1, :], converted_hor[2, :])

    converted_ver = ConvertToXYZ(dataframe, bunny, "vertical")
    converted_ver = CorrectAxes(converted_ver)
    converted_ver = point_coords(converted_ver[0, :], converted_ver[1, :], converted_ver[2, :])

    converted_dataframe = plane_dataframe(converted_til, converted_hor, converted_ver)

    return converted_dataframe

def AffineFit(nominal_plane_pts):
    centre = np.mean(nominal_plane_pts, axis = 1) # Axes are: X, Y, Z
    delta = nominal_plane_pts.T - centre
    delta = delta.astype(np.float)
    eigen_vector = np.linalg.svd(delta.T)
    left_col = eigen_vector[0]
    normal = left_col[:, -1]

    return centre, normal

def VerticalAxesSwitch(ver_bunny_pts, ver_plane_pts):
    rotated_bunny_pts, rotated_plane_pts = np.zeros(ver_bunny_pts.shape), np.zeros(ver_plane_pts.shape)
    rotated_bunny_pts[0, :], rotated_plane_pts[0, :] = ver_bunny_pts[2, :], ver_plane_pts[2, :]
    rotated_bunny_pts[1, :], rotated_plane_pts[1, :] = ver_bunny_pts[1, :], ver_plane_pts[1, :]
    rotated_bunny_pts[2, :], rotated_plane_pts[2, :] = ver_bunny_pts[0, :], ver_plane_pts[0, :]

    return (rotated_bunny_pts, rotated_plane_pts)

def CreateContours(tmp_bunny_pts, nominal_plane_pts, plane_tolerance, plane):
    if plane.startswith('ver'):
        tmp_bunny_pts, nominal_plane_pts = VerticalAxesSwitch(tmp_bunny_pts, nominal_plane_pts)

    centre, normal = AffineFit(nominal_plane_pts)

    contour_pts = np.zeros((3, 1))
    for idx in range(tmp_bunny_pts.shape[1]):
        delta = abs(np.dot(tmp_bunny_pts[:, idx] - centre, normal))
        if delta <= plane_tolerance:
            contour_pts = np.append(contour_pts, tmp_bunny_pts[:, idx].reshape(3, 1), axis = 1)

    if plane.startswith('ver'):
        contour_pts, _ = VerticalAxesSwitch(contour_pts, nominal_plane_pts)

    return contour_pts[:, 1:]

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
