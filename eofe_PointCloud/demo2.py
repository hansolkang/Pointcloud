import numpy as np
import os
import pandas as pd
import copy
import open3d as o3d
from tkinter import filedialog
from tkinter import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class PCDATA(object):
    def __init__(self,fid):
        self.fid = fid
        self.data = pd.read_csv(fid, header=None)

        self.data[0] = self.data[0].str[1:]
        self.data[0] = self.data[0].astype('float64')
        # self.rkreh.append(self.data[1])
        self.data[1] = -self.data[1]
        self.data[2] = -self.data[2]

        # self.interpolation(fid,2,2,output_file)
        for a in range(1, 6):
            self.data[a] = self.data[a] * np.pi / 180
        # print(self.data.head())
        self.position=[[],[],[]]
        self.itposition = [[], [], []]

    def transform2d(self):
        scdata = self.data
        for i,r in enumerate(self.data[0]):
            phi = self.data[1][i]
            theta = self.data[2][i]
            head = self.data[3][i]
            beta = self.data[4][i]
            alpha = self.data[5][i]
            pst = self.transform2d_(r,head,theta,phi,alpha,beta)
            self.position[0].append(pst[0][0])
            self.position[1].append(pst[1][0])
            self.position[2].append(pst[2][0])

    @staticmethod
    def transform2d_(r, head, theta, phi, alpha, beta):
        arr = np.array([[r], [0], [0]])
        trans0 = np.array([[np.cos(head), np.sin(head), 0], [-np.sin(head), np.cos(head), 0], [0, 0, 1]])
        trans1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
        trans2 = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
        trans3 = np.array([[np.cos(phi), np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        trans4 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])

        # position = trans0.dot(trans1).dot(trans2).dot(trans3).dot(trans4).dot(arr)
        position = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(trans0, trans1), trans2), trans3), trans4), arr)
        return position

    def adtransform(self, inputposition, x, y, z, theta, phi, roll):
        trans0 = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        trans1 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        trans2 = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])

        return trans0.dot(trans1).dot(trans2).dot(inputposition) + np.array([[x], [y], [z]])

    def interpolation01(self, vertical_ratio=4):
        array1 = np.zeros([3600, 3])
        array2 = np.zeros([3600, 3])

        self.data[1] = -self.data[1]
        self.data[2] = -self.data[2]

        for i in range(1, 6):
            self.data[i] = self.data[i] / np.pi * 180

        t1 = self.data[2][0]
        ext = t1 + 0
        for i, r in enumerate(self.data[0]):
            p2 = self.data[1][i]
            t2 = self.data[2][i]
            h2 = self.data[3][i]
            b2 = self.data[4][i]
            a2 = self.data[5][i]
            dt = t2 - t1

            if dt == 0:
                array2[int(p2 * 10)][:] = [r, b2, a2]

            else:
                num_zero = 0
                exr = np.zeros([3])
                for ii in range(3601):
                    r2 = array2[ii % 3600, :]
                    if r2[0] > 0:
                        if num_zero < 30:
                            if abs(r2[0] - exr[0]) < 0.1 * min(r2[0], exr[0]):
                                for hi in range(ii - num_zero, ii + 1):
                                    ir = ((r2 - exr) / (num_zero + 1) * (hi - (ii - num_zero - 1)) + exr)
                                    array2[(hi % 3600), :] = ir
                        num_zero = 0
                        exr[:] = r2
                    else:
                        num_zero += 1

                for ii in range(3600):

                    r1 = array1[ii, :]
                    r2 = array2[ii, :]

                    pst = self.transform2d_(r2[0], h2 * np.pi / 180, -t1 * np.pi / 180, -ii * 0.1 * np.pi / 180,
                                            r2[2] * np.pi / 180, r2[1] * np.pi / 180)
                    self.itposition[0].append(pst[0][0])
                    self.itposition[1].append(pst[1][0])
                    self.itposition[2].append(pst[2][0])

                    if (r1[0] > 0) & (r2[0] > 0):

                        if abs(r2[0] - r1[0]) < (0.1 * min(r2[0], r1[0])):

                            for vi in range(1, vertical_ratio):
                                ir = (1 - vi / vertical_ratio) * r1 + vi / vertical_ratio * r2
                                it = (1 - vi / vertical_ratio) * ext + vi / vertical_ratio * t1
                                ip = ii * 0.1

                                pst = self.transform2d_(ir[0], h2 * np.pi / 180, -it * np.pi / 180, -ip * np.pi / 180,
                                                        ir[2] * np.pi / 180, ir[1] * np.pi / 180)
                                self.itposition[0].append(pst[0][0])
                                self.itposition[1].append(pst[1][0])
                                self.itposition[2].append(pst[2][0])

                ext = t1 + 0
                array1[:] = array2[:]
                array2 = np.zeros([3600, 3])

            t1 = t2 + 0

        self.data[1] = -self.data[1]
        self.data[2] = -self.data[2]

        for i in range(1, 6):
            self.data[i] = self.data[i] * np.pi / 180

        itpcd = o3d.geometry.PointCloud()
        itpcd.points = o3d.utility.Vector3dVector(np.array(self.itposition).T)
        o3d.io.write_point_cloud(self.fid[:-4] + 'interp.ply', itpcd)
        return itpcd

def file_finder():
    file = filedialog.askopenfilenames(initialdir="D:\hansol\Open3D2\examples\Python\Advanced\1016")
    return file

def draw_registration_result_and_write(source, target, transformation, directory, result_file):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    print(source_temp.PointCloud)
    if result_file == 'result.ply':
        o3d.visualization.draw_geometries([source_temp, target_temp])
    source.transform(transformation)
    source += target
    temp = o3d.geometry.PointCloud()
    temp =source
    o3d.io.write_point_cloud(directory + result_file, temp, True);


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    print(source_temp.PointCloud)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def crop_geometry_1(directory):
    pcd = o3d.io.read_point_cloud(directory + "eofe0.ply")
    o3d.visualization.draw_geometries_with_editing([pcd])

def crop_geometry_2(directory):
    pcd = o3d.io.read_point_cloud(directory + "eofe1.ply")
    o3d.visualization.draw_geometries_with_editing([pcd])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=20))
    # pcd_fpfh = compute_fpfh_feature(pcd_down, KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 20))
    return pcd_down, pcd_fpfh

def prepare_dataset_1(voxel_size, directory):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(directory + "cropped_1.ply")
    target = o3d.io.read_point_cloud(directory + "cropped_2.ply")

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset_2(voxel_size, directory):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(directory + "eofe0.ply")
    target = o3d.io.read_point_cloud(directory + "eofe1.ply")

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)


    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def prepare_dataset_3(voxel_size, file1, file2):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(file1)
    target = o3d.io.read_point_cloud(file2)

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)


    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def display_inlier_outlier(cloud, ind, directory):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (blue): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.io.write_point_cloud(directory + "sta_removal.ply", inlier_cloud)

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2
    # distance_threshold = voxel_size * 5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,distance_threshold,o3d.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.01),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(8000000, 1000))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    # distance_threshold = voxel_size * 0.3
    distance_threshold = voxel_size * 0.5
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,o3d.registration.TransformationEstimationPointToPlane())
    return result
def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False


if __name__ == "__main__":
    array = []

    root = Tk()
    root.update()
    for i in range(2):
        a = file_finder()
        a = ''.join(a)
        array.append(a)

    root.destroy()
    directory = a[:-24]

    key_to_callback = {}
    threshold = 0.5
    voxel_size_1 = 0.15  # means 5cm for the dataset

    for i in range(2):
        # a = "C:\Users\hci\Desktop\Open3D\examples\Python\Basic(0.8)\0826_data\PUSAN_1\DASS_20190826_135404.txt"
        print("Converting TEXT to PLY...")

        fids_data = PCDATA(array[i])
        fids_data.transform2d()
        output = fids_data.position       #original

        list = copy.deepcopy(output[2])
        output2_min = min(output[2])
        output2_max = max(output[2])

        index_min = list.index(output2_min)
        index_max = list.index(output2_max)

        for pop_i in range(3):
            output[pop_i].pop(index_min)
            output[pop_i].pop(index_max)
        minus_output2_min = min(output[2])
        minus_output2_max = max(output[2])


        file = open(directory + 'eofe' + str(i) + '.ply', mode='wt')
        file.write('ply\n' + 'format ascii 1.0\n' + 'comment Created by hansol\n' + 'element vertex '
                   + str(len(output[0])-1) + '\nproperty double x\n'+ 'property double y\n' + 'property double z\n'
                   + 'property uchar red\n' + 'property uchar green\n' + 'property uchar blue\n' + 'end_header\n')

        for index in range(len(output[0])):
            z_color_r = 0
            # z_color_r = int(abs((output[2][index] - min(output[2])) / (max(output[2]) - min(output[2]))) * 255)
            z_color_r = int(abs((output[2][index] - minus_output2_min) / (minus_output2_max - minus_output2_min)) * 255)
            file.write(str(output[0][index]) + ' ' + str(output[1][index]) + ' ' + str(output[2][index])+
                       ' 200 ' + str(z_color_r) + ' 0' + '\n')
            file.flush()
        file.close


        if os.path.exists(array[1][:-4]+'interp.ply') == False:
            print("Interpolation...")
            fids_data.interpolation01()

    print("Crop Subset")
    crop_geometry_1(directory)
    crop_geometry_2(directory)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_1(voxel_size_1,directory)

    print("Result_ransac : Subset global registration")
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size_1)

    print("Crop subset: Subset ICP registration")
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size_1)
    subset_trans = result_icp.transformation
    # draw_registration_result(source, target, subset_trans)

    print("Refine_registration : PointToPint Transformation")
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, subset_trans, o3d.registration.TransformationEstimationPointToPoint())
    # draw_registration_result(source, target, reg_p2p.transformation)

    print("Whole data read")
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_2(voxel_size_1,directory)

    print("Whole data transform")
    draw_registration_result_and_write(source, target, reg_p2p.transformation, directory, 'result.ply')

    print("Interpolation data read")
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset_3(voxel_size_1, array[0][:-4] + 'interp.ply', array[1][:-4] + 'interp.ply')

    print("Interpolation data transform")
    draw_registration_result_and_write(source, target, reg_p2p.transformation, directory, "Inter_result.ply")

    print("K key is black background")
    pcd = o3d.io.read_point_cloud(directory + 'result.ply')
    key_to_callback[ord("K")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    while(True):
        print("1.Visualize             2.Interpolation Visualization           ")
        print("3.Remove outlier        4.Outlier Removal Visualization     0. Quit")
        print("")
        userinput = int(input("Select Number : "))
        if userinput == 1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            result_file = directory + "result.ply"
            x, y, z = [], [], []

            with open(result_file) as f:
                for index, line in enumerate(f):
                    if index <= 11:
                        continue
                    else:
                        array_line = line.split()
                        x.append(array_line[0])
                        y.append(array_line[1])
                        z.append(array_line[2])
            x = map(float, x)
            y = map(float, y)
            z = map(float, z)
            result = ax.scatter(x, y, z, c=z, s=10, alpha=1, cmap=plt.cm.rainbow, label=None)
            ax.set_xlabel('x_label')
            ax.set_ylabel('y_label')
            ax.set_zlabel('z_label')
            fig.colorbar(result)

            plt.show()
            # print("K key is black background")
            # pcd = o3d.io.read_point_cloud(directory + 'result.ply')
            # key_to_callback[ord("K")] = change_background_to_black
            # o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

        elif userinput == 2:
            print("K key is black background")
            pcd = o3d.io.read_point_cloud(directory + 'Inter_result.ply')
            key_to_callback[ord("K")] = change_background_to_black
            o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
        elif userinput == 3:
            print("Outlier removing...")
            pcd = o3d.io.read_point_cloud(directory + 'result.ply')
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
            display_inlier_outlier(pcd, ind, directory)
        elif userinput == 4:
            print("K key is black background")
            pcd = o3d.io.read_point_cloud(directory + 'sta_removal.ply')
            key_to_callback[ord("K")] = change_background_to_black
            o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
        elif userinput == 0:
            break



print("end")