import open3d as o3d
import numpy as np
import sys

if __name__ == "__main__":
    # points = [[-0.16695494, -1.2914611, 0.], [-1.52347555, -0.57049949, 0.],
    #           [-0.64916301, -0.87521039, 0.], [1.41453302, 1.66485933, 0.],
    #           [1.99241562, -1.56264091, 0.], [0.14994255, 0.65654215, 0.],
    #           [-0.33633312, -0.3207582, 0.], [-0.29291572, -0.04766658, 0.],
    #           [-0.04255177, 1.87292308, 0.]]
    # normals = [[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
    #            [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.],
    #            [0., 0., 1.]]
    radii = [0.1, 1]
    # pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/DASS_20190920_165420xyz.ply")
    pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/sta_removal_result_reg.ply")
    # pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/result_reg.ply")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,max_nn=40),fast_normal_computation=True)
    # o3d.visualization.draw_geometries([pcd])
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([pcd])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([pcd, mesh])