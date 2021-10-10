#!/usr/bin/evn python3
import open3d as o3d

def LoadMesh(data_dir, bunny_name):
    print("Loading meshes in open3d...")
    mesh = o3d.io.read_triangle_mesh(data_dir + "/" + bunny_name + '.stl')

    return mesh

def LoadPointcloud(data_dir, pcd_name):
    print("Loading meshes in open3d...")
    pcd1 = o3d.io.read_point_cloud(data_dir + "/" + pcd_name + ".pcd")

    return pcd1

def PointcloudFromMesh(data_dir, mesh, meshname):
    print("Generating point cloud from mesh model...")
    pcd = mesh.sample_points_poisson_disk(75000)
    print("Saving PointCloud to " + data_dir + meshname + ".pcd")
    o3d.io.write_point_cloud(data_dir + meshname + ".pcd", pcd)

def MeshToPointcloud(bunny_mesh_name, bunny_pcd_name):
    bunny_mesh = LoadMesh(bunny_mesh_name)
    PointcloudFromMesh(bunny_mesh, bunny_pcd_name)
    bunny_pcd = LoadPointcloud(bunny_pcd_name)
    o3d.visualization.draw_geometries([bunny_pcd])

def GeneratePointclouds():
    for bunny_idx in range(1, 5):
        bunny_mesh_name = "bunny_" + str(bunny_idx) + "_new"
        bunny_pcd_name = "bunny_" + str(bunny_idx) + "_pcd"
        MeshToPointcloud(bunny_mesh_name, bunny_pcd_name)

def DisplayAlignedModels(anchor, free, transform):
    anchor_temp = copy.deepcopy(anchor)
    free_temp = copy.deepcopy(free)

    anchor_temp.paint_uniform_color([0, 0.7, 0.8])
    free_temp.paint_uniform_color([0.8, 0.7, 0])

    free_temp.transform(transform)

    o3d.visualization.draw_geometries([free_temp, anchor_temp])