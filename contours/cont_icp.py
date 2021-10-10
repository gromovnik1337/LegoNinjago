#!/usr/bin/evn python3
import copy
import numpy as np
import open3d as o3d
from cont_utlis_icp import LoadMesh, LoadPointcloud, PointcloudFromMesh, MeshToPointcloud, GeneratePointclouds, DisplayAlignedModels

# Append the required sys.path for accessing the other directories
import os
import sys
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir("..")
data_dir = os.getcwd() + "/data"
sys.path.append(data_dir)
os.chdir(curr_dir)

def AlignmentProcedure(data_dir, cad_pcd_name, scan_pcd_name, plot):

    # Load the clouds
    cad_pcd = LoadPointcloud(data_dir, cad_pcd_name)
    scan_pcd = LoadPointcloud(data_dir, scan_pcd_name)

    threshold = 2
    initial_transformation = np.diagflat([1, 1, 1, 1]) # scaling in each direction (X, Y, Z) with overall scaling as a 4th dimension (i.e. scales all 3 dimensions)
    initial_transformation[0:3,3] = [0, 0, 0] # [X, Y, Z] initial alignment estimated adjustments
    scan_pcd.transform(initial_transformation)

    cad_np_pcd = np.asarray(cad_pcd.points)
    cad_np_pcd = cad_np_pcd[np.where(cad_np_pcd[:, 0] < 27)]

    cropped_cad_pcd = o3d.geometry.PointCloud()
    cropped_cad_pcd.points = o3d.utility.Vector3dVector(cad_np_pcd)
    cropped_cad_pcd.estimate_normals()

    scan_np_pcd = np.asarray(scan_pcd.points)
    scan_np_pcd = scan_np_pcd[np.where(scan_np_pcd[:, 0] < 27)]

    cropped_scan_pcd = o3d.geometry.PointCloud()
    cropped_scan_pcd.points = o3d.utility.Vector3dVector(scan_np_pcd)
    cropped_scan_pcd.estimate_normals()

    if plot:
        cad_pcd.paint_uniform_color([0, 0.7, 0.8])
        scan_pcd.paint_uniform_color([0.8, 0.7, 0])
        o3d.visualization.draw_geometries([cad_pcd, scan_pcd])
    else:
        pass

    initial_transformation = np.diagflat([1, 1, 1, 1])
    evaluation = o3d.pipelines.registration.evaluate_registration(cropped_cad_pcd, cropped_scan_pcd, threshold, initial_transformation)

    print("Applying ICP alignment to point clouds...")

    reg_p2p = o3d.pipelines.registration.registration_icp(source = cropped_scan_pcd, target = cropped_cad_pcd,\
    max_correspondence_distance = threshold, init = initial_transformation,\
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),\
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-6, relative_rmse = 1e-6, max_iteration = 500))

    final_transformation = reg_p2p.transformation
    print("Transformation is: ")
    print(final_transformation)

    evaluation = o3d.pipelines.registration.evaluate_registration(scan_pcd, cad_pcd, threshold, final_transformation)
    print(evaluation)

    if plot:
        scan_pcd.transform(final_transformation)
        cad_pcd.paint_uniform_color([0, 0.7, 0.8])
        scan_pcd.paint_uniform_color([0.8, 0.7, 0])
        o3d.visualization.draw_geometries([cad_pcd, scan_pcd])

    else:
        pass

    '''
    print("Saving Repositioned CAD Pointcloud...")
    o3d.io.write_point_cloud(data_dir + cad_pcd_name + ".pcd", cad_pcd)


    print("Saving Aligned PointCloud...")
    o3d.io.write_point_cloud(data_dir + scan_pcd_name + ".pcd", scan_pcd)
    '''

def IterativeAlignment():
    for bunny_idx in range(1, 5):
        AlignmentProcedure(data_dir, "cad_bunny_pcd", "bunny_" + str(bunny_idx) + "_pcd", True)

if __name__ == '__main__':
    IterativeAlignment()
