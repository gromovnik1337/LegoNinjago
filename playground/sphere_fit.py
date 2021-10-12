import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sqrt
import os
import open3d as o3d
import pandas as pd

def loadPointcloud(data_dir, pcd_name):
    print("Loading meshes in open3d...")
    pcd1 = o3d.io.read_point_cloud(data_dir + "/" + pcd_name + ".pcd")

    return pcd1

def setAxesEqual(ax):
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

def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = sqrt(t)

    return radius, C[0], C[1], C[2]

def generateSphere(centres, radii, segments):
    x_c, y_c, z_c = centres
    x_r, y_r, z_r = radii

    phi, theta = np.meshgrid(np.linspace(0, pi * 2, segments), np.linspace(0, pi * 2, segments))
    xx = x_c + (x_r * np.sin(theta) * np.cos(phi))
    yy = y_c + (y_r * np.sin(theta) * np.sin(phi))
    zz = z_c + (z_r * np.cos(theta))

    return xx.flatten(), yy.flatten(), zz.flatten()

curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"

n_segs = 71

sphere_centres  =   (12, 12, 12)
sphere_radii    =   (6, 6, 5)
sphere_x, sphere_y, sphere_z = generateSphere(sphere_centres, sphere_radii, n_segs)

print(sphere_x.shape, sphere_y.shape, sphere_z.shape)

radius, centre_x, centre_y, centre_z = sphereFit(sphere_x, sphere_y, sphere_z)
fitted_x, fitted_y, fitted_z = generateSphere((centre_x, centre_y, centre_z), (radius, radius, radius), n_segs)


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(X = sphere_x.reshape(n_segs, n_segs), Y = sphere_y.reshape(n_segs, n_segs), Z = sphere_z.reshape(n_segs, n_segs), color = 'red', alpha = 0.4)
ax.plot_wireframe(X = fitted_x.reshape(n_segs, n_segs), Y = fitted_y.reshape(n_segs, n_segs), Z = fitted_z.reshape(n_segs, n_segs), color = 'blue', alpha = 0.6)
setAxesEqual(ax)
plt.show()


cloud = loadPointcloud(data_dir, "bunny_1_pcd")
cloud = np.array(cloud.points)

cloud_pd = pd.DataFrame(cloud)
cloud_pd.columns = ['X', 'Y', 'Z']
cloud_pd["function"] = (cloud_pd["X"] * sqrt(2) / 2 + cloud_pd["Y"] * sqrt(2) / 2 + cloud_pd["Z"] * sqrt(2) / 2)
cloud_pd = cloud_pd.sort_values(["function"])
print(cloud_pd)
cloud = cloud_pd.to_numpy()

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(xs = cloud[0:35000:100, 0], ys = cloud[0:35000:100, 1], zs = cloud[0:35000:100, 2], color = 'blue', alpha = 0.6)
setAxesEqual(ax)
plt.show()

'''
xx, yy, zz = np.meshgrid(cloud[::100, 0], cloud[::100, 1], cloud[::100, 2])
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(X = xx, Y = yy, Z = zz, color = 'red', alpha = 0.4)
setAxesEqual(ax)
plt.show()
'''
