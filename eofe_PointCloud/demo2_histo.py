import numpy as np
import os
import pandas as pd
import copy
import open3d as o3d
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

class PCDATA(object):
    def __init__(self,fid):
        self.fid = fid
        self.data = pd.read_csv(fid, header=None)

        # self.data[0] = self.data[0].str[1:]
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
def delete_zero(input_file, index):
    f = open(input_file, mode = 'r')
    lines = f.readlines()
    result = []

    for line in lines:
        if not "$0.0000" in line:
            result.append(line[1:])
        f.close()
    f = open("./0920/zero_delete" + str(index) + ".txt", 'w')
    f.writelines(result)
    f.close()

if __name__ == "__main__":
    first = []

    first.append("./0920/DASS_20190920_165420.txt")
    first.append("./0920/DASS_20190920_170724.txt")

    threshold = 0.5
    voxel_size_1 = 0.15  # means 5cm for the dataset
    X = []
    Y = 0
    while (True):
        for i in range(2):
            print("Converting TEXT to zero_delete...")
            if not os.path.isfile("./0920/zero_delete0.txt"):
                delete_zero(first[i], i)
        fids_data = PCDATA("./0920/zero_delete0.txt")
        # fids_data = PCDATA(array[i])
        fids_data.transform2d()
        output = fids_data.position       #original

        X[i] = output[2]
        hist = plt.hist(X, bins=255, density=False, cumulative=False, label='A')
                        # range=(X.min() - 1, X.max() + 1), color='r', edgecolor='black', linewidth=1.2)

        plt.title('scatter', pad=10)
        plt.xlabel('X axis', labelpad=10)
        plt.ylabel('Y axis', labelpad=20)

        plt.minorticks_on()
        plt.tick_params(axis='both', which='both', direction='in', pad=8, top=True, right=True)

        plt.show()
            # file = open(directory + 'eofe' + str(i) + '.ply', mode='wt')
            # file.write('ply\n' + 'format ascii 1.0\n' + 'comment Created by hansol\n' + 'element vertex '
            #            + str(len(output[0])-1) + '\nproperty double x\n'+ 'property double y\n' + 'property double z\n'
            #            + 'property uchar red\n' + 'property uchar green\n' + 'property uchar blue\n' + 'end_header\n')
            #
            # for index in range(len(output[0])):
            #     z_color_r = 0
            #     z_color_r = int(abs((output[2][index] - minus_output2_min) / (minus_output2_max - minus_output2_min)) * 255)
            #     file.write(str(output[0][index]) + ' ' + str(output[1][index]) + ' ' + str(output[2][index])+
            #                ' 200 ' + str(z_color_r) + ' 0' + '\n')
            #     file.flush()
            # file.close


    #
    # print("Crop Subset")
    # crop_geometry_1(directory)
    # crop_geometry_2(directory)
    # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_1(voxel_size_1,directory)
    #
    # print("Result_ransac : Subset global registration")
    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size_1)
    #
    # print("Crop subset: Subset ICP registration")
    # result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size_1)
    # subset_trans = result_icp.transformation
    # # draw_registration_result(source, target, subset_trans)
    #
    # print("Refine_registration : PointToPint Transformation")
    # reg_p2p = o3d.registration.registration_icp(source, target, threshold, subset_trans, o3d.registration.TransformationEstimationPointToPoint())
    # # draw_registration_result(source, target, reg_p2p.transformation)
    #
    # print("Whole data read")
    # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_2(voxel_size_1,directory)
    #
    # print("Whole data transform")
    # draw_registration_result_and_write(source, target, reg_p2p.transformation, directory, 'result.ply')
    #
    # print("Interpolation data read")
    # source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #     prepare_dataset_3(voxel_size_1, array[0][:-4] + 'interp.ply', array[1][:-4] + 'interp.ply')
    #
    # print("Interpolation data transform")
    # draw_registration_result_and_write(source, target, reg_p2p.transformation, directory, "Inter_result.ply")
    #
    # print("K key is black background")
    # pcd = o3d.io.read_point_cloud(directory + 'result.ply')
    # key_to_callback[ord("K")] = change_background_to_black
    # o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    # while(True):
    #     print("1.Visualize             2.Interpolation Visualization           ")
    #     print("3.Remove outlier        4.Outlier Removal Visualization     0. Quit")
    #     print("")
    #     userinput = int(input("Select Number : "))
    #     if userinput == 1:
    #         print("K key is black background")
    #         pcd = o3d.io.read_point_cloud(directory + 'result.ply')
    #         key_to_callback[ord("K")] = change_background_to_black
    #         o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    #         # print("Plus Mesh")
    #         # radii = [0.1, 1]
    #         # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=40),fast_normal_computation=True)
    #         # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #         #     pcd, o3d.utility.DoubleVector(radii))
    #         # mesh.compute_vertex_normals()
    #         # o3d.visualization.draw_geometries([pcd, mesh])
    #
    #     elif userinput == 2:
    #         print("K key is black background")
    #         pcd = o3d.io.read_point_cloud(directory + 'Inter_result.ply')
    #         key_to_callback[ord("K")] = change_background_to_black
    #         o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    #     elif userinput == 3:
    #         print("Outlier removing...")
    #         pcd = o3d.io.read_point_cloud(directory + 'result.ply')
    #         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
    #         display_inlier_outlier(pcd, ind, directory)
    #     elif userinput == 4:
    #         print("K key is black background")
    #         pcd = o3d.io.read_point_cloud(directory + 'sta_removal.ply')
    #         key_to_callback[ord("K")] = change_background_to_black
    #         o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    #     elif userinput == 0:
    #         break



print("end")