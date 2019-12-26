import open3d as o3d

import trimesh

import numpy as np

# pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/hansol1.ply")
pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/result_reg.ply")
pcd.estimate_normals()

# estimate radius for rolling ball

distances = pcd.compute_nearest_neighbor_distance()

avg_dist = np.mean(distances)

radius = 1.5 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 5]))

trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),vertex_normals=np.asarray(mesh.vertex_normals))

o3d.visualization.draw_geometries([pcd, mesh])