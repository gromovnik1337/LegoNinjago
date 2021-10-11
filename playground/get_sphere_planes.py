#!/usr/bin/evn python3
import copy
import numpy as np
import open3d as o3d
# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
cont_dir = os.getcwd() + "/contours"
sys.path.append(data_dir)
sys.path.append(cont_dir)
os.chdir(curr_dir)

from cont_utlis_icp import LoadMesh, LoadPointcloud, PointcloudFromMesh, MeshToPointcloud, GeneratePointclouds, DisplayAlignedModels

# Load the cloud
pcd_name = "bunny_1_pcd"
pcd = LoadPointcloud(data_dir, pcd_name)

pcd.paint_uniform_color([0.8, 0.7, 0])
o3d.visualization.draw_geometries([pcd])