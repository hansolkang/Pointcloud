# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/outlier_removal.py

import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (blue): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    # o3d.io.write_point_cloud("../../TestData/19_10_13/sta_removal_0920.ply", inlier_cloud)\
    # o3d.io.write_point_cloud("../../TestData/19_10_13/sta_removal_0922672.ply", inlier_cloud)


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/result_xyz.ply")
    # pcd = o3d.io.read_point_cloud("../../TestData/19_10_13/DASS_20190826_183918xyz.ply")
    pcd = o3d.io.read_point_cloud("../1016/result.ply")
    # o3d.visualization.draw_geometries([pcd])

    # print("Downsample the point cloud with a voxel of 0.01")
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # o3d.visualization.draw_geometries([voxel_down_pcd])

    # print("Every 5th points are selected")
    # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # o3d.visualization.draw_geometries([uni_down_pcd])

    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.6)
    display_inlier_outlier(pcd, ind)

    # print("Radius oulier removal")
    # cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=1)
    # display_inlier_outlier(pcd, ind)
