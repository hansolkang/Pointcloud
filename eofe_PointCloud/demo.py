import numpy as np
from numpy import median
import glob
import pandas as pd
import copy
import open3d as o3d
from tkinter import filedialog
from tkinter import *

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

    def transform2d_(r, head, theta, phi, alpha, beta):
        arr = np.array([[r], [0], [0]])
        trans0 = np.array([[np.cos(head), np.sin(head), 0], [-np.sin(head), np.cos(head), 0], [0, 0, 1]])  # z축
        trans1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])  # x축
        trans2 = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])  # y축
        trans3 = np.array([[np.cos(phi), np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])  # z축
        trans4 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])  # y축

        # position = trans0.dot(trans1).dot(trans2).dot(trans3).dot(trans4).dot(arr)
        position = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(trans0, trans1), trans2), trans3), trans4), arr)
        return position

    def transform2d_(self, r, head, theta, phi, alpha, beta):
        # r: distance
        # head: head
        # theta: horizontal
        # phi: vertical
        # alpha: roll
        # beta: pitch

        arr = np.array([[r],[0],[0]])
        trans0 = np.array([[np.cos(head), np.sin(head), 0],[-np.sin(head), np.cos(head), 0],[0, 0, 1]]) #z
        trans1 = np.array([[1, 0, 0],[0, np.cos(alpha), np.sin(alpha)],[0, -np.sin(alpha), np.cos(alpha)]]) #x
        trans2 = np.array([[np.cos(beta), 0, -np.sin(beta)],[0, 1, 0],[np.sin(beta), 0, np.cos(beta)]]) #y
        trans3 = np.array([[np.cos(phi), np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0, 0, 1]]) #z
        trans4 = np.array([[np.cos(theta), 0, -np.sin(theta)],[0, 1, 0],[np.sin(theta), 0, np.cos(theta)]]) #y

        position = trans0.dot(trans1).dot(trans2).dot(trans3).dot(trans4).dot(arr)
        return position

def file_finder():
    file = filedialog.askopenfilenames(initialdir="C:\Users\hci\Desktop\Open3D\examples\Python\Basic(0.8)")
    return file

def delete_zero(input_file):
    f = open(input_file, mode = 'r')
    lines = f.readlines()
    result = []

    for line in lines:
        if not "$0.0000" in line:
            result.append(line[1:])
        f.close()
    f = open(directory + "zero_delete.txt", 'w')
    f.writelines(result)
    f.close()
    f = open(directory + "zero_delete.txt",'r')
    return f

def horizontal_interpolation(input_file, horizontal, output_file):
    f = delete_zero(input_file)
    f1 = open(output_file, mode='w')
    lines = f.readlines()
    i, j = 0, 1

    for line in lines:
        if j == len(lines) - 1: break
        asdf = line[:][:-1].split(',')
        asdf = float(asdf[0])
        if asdf >=2.0:
            f1.writelines(line)
            inter_calcul = []
            result = ''
            first = lines[i][:][:-1].split(',')
            second = lines[j][:][:-1].split(',')
            if float(first[0]) - float(second[0]) <= 5:
                for a in range(0, 6):
                    inter_calcul.append((float(first[a]) + float(second[a])) / 2)
                    result += str(inter_calcul[a])
                    if a != 5:
                        result += ', '
                f1.writelines(result + '\n')
        i += 1
        j += 1
    f.close()
    f1.close()
def vertical_interpolation(input_file, vertical, output_file):
    # f = self.delete_zero(input_file)
    f = open(input_file, mode = 'r')
    f1 = open(output_file, mode = 'w')
    lines = f.readlines()
    i,j=0,1

    for index in range(len(lines) - 1):
        a = lines[index][:][:-1].split(',')
        lines[index] = map(float, a)
    lines.sort(key=lambda x: x[1])
    for line in lines:
        if j == len(lines)-1 : break
        result = '$'
        for a in range(0,6):
            result += str(line[a])
            if a != 5:
                result += ', '
        f1.writelines(result + '\n')
        inter_calcul = []
        result = '$'
        first = lines[i][:][:]
        second = lines[j][:][:]
        for a in range(0,6):
            inter_calcul.append((float(first[a]) + float(second[a]))/2)
            result += str(inter_calcul[a])
            if a != 5:
                result += ', '
        f1.writelines(result + '\n')
        i += 1
        j += 1
    f.close()
    f1.close()

def draw_registration_result_and_write(source, target, transformation, directory):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    print(source_temp.PointCloud)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    source.transform(transformation)
    source += target
    temp = o3d.geometry.PointCloud()
    temp =source
    o3d.io.write_point_cloud(directory + 'result.ply', temp, True);


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
    source = o3d.io.read_point_cloud(directory + "crop0.ply")
    target = o3d.io.read_point_cloud(directory + "crop1.ply")

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

    # source = read_point_cloud("../Basic/0826_data/09201/10161n_notmul.pcd")
    # target = read_point_cloud("../Basic/0826_data/09201/10162n_notmul_2.pcd")

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))


    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


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

def multiple_vis():
    pcd1 = o3d.io.read_point_cloud(directory + "eofe0.ply")
    pcd2 = o3d.io.read_point_cloud(directory + "eofe1.ply")
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window()
    vis2.create_window()
    vis1.add_geometry(pcd1)
    vis2.add_geometry(pcd2)
    vis1.run()  # user picks points
    vis2.run()  # user picks points
    vis1.destroy_window()
    vis2.destroy_window()

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run() # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def manual_registration(directory):
    source = o3d.io.read_point_cloud(directory + "eofe0.ply")
    target = o3d.io.read_point_cloud(directory + "eofe1.ply")

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)

    assert(len(picked_id_source)>=3 and len(picked_id_target)>=3)
    assert(len(picked_id_source) == len(picked_id_target))

    corr = np.zeros((len(picked_id_source),2))
    corr[:,0] = picked_id_source
    corr[:,1] = picked_id_target

    print("Transformation Estimation Point To Point")
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    print("Perform point-to-point ICP refinement")
    threshold = 0.05 # 3cm distance threshold
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == "__main__":
    array = []
    for i in range(2):
        a = file_finder()
        a = ''.join(a)
        array.append(a)
    directory = a[:-24]
    horizontal_output_file = directory + "horizontal_interpol.txt"
    vertical_output_file = directory + "vertical_interpol.txt"

    key_to_callback = {}
    threshold = 0.5
    voxel_size_1 = 0.15  # means 5cm for the dataset

    while(True):
        for i in range(2):
            # a = "C:\Users\hci\Desktop\Open3D\examples\Python\Basic(0.8)\0826_data\PUSAN_1\DASS_20190826_135404.txt"
            print("Converting TEXT to PLY...")
            fids_data = PCDATA(array[i])
            fids_data.transform2d()
            output = fids_data.position
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

            mide = np.sort(list)[::-1][int(0.15 * len(list))]
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

        print("Crop Subset")
        crop_geometry_1(directory)
        crop_geometry_2(directory)
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_1(voxel_size_1,directory)

        print("Result_ransac : Subset global registration")
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size_1)

        print("Crop subset: Subset ICP registration")
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size_1)
        subset_trans = result_icp.transformation
        draw_registration_result(source, target, subset_trans)

        print("Refine_registration : PointToPint Transformation")
        reg_p2p = o3d.registration.registration_icp(source, target, threshold, subset_trans, o3d.registration.TransformationEstimationPointToPoint())
        draw_registration_result(source, target, reg_p2p.transformation)

        print("Whole data read")
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset_2(voxel_size_1,directory)

        print("Whole data transform")
        draw_registration_result_and_write(source, target, reg_p2p.transformation, directory)
        # evaluation = evaluate_registration(source, target, threshold, subset_trans)

        print("K key is black background")
        pcd = o3d.io.read_point_cloud(directory + 'result.ply')
        key_to_callback[ord("K")] = change_background_to_black
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

        print("1.Interpolation              2. Plus Mesh         3. Remove outlier       4. Visualize")
        print("5. Select Feature Point      0. Quit")
        print("")
        userinput = input("Select Number : ")
        if userinput == 1:
            for i in range(2):
                horizontal_interpolation(array[i], 2, horizontal_output_file)
                vertical_interpolation(horizontal_output_file, 2, vertical_output_file)
                real_output_directory = glob.glob(directory + 'vertical_interpol.txt')
                fids_data = PCDATA(real_output_directory[0])
                fids_data.transform2d()
                output = fids_data.position
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

                mide = np.sort(list)[::-1][int(0.15 * len(list))]
                file = open(directory + 'inter_eofe' + str(i) + '.ply', mode='wt')
                file.write('ply\n' + 'format ascii 1.0\n' + 'comment Created by hansol\n' + 'element vertex '
                           + str(len(output[0])-1) + '\nproperty double x\n'+ 'property double y\n' + 'property double z\n'
                           + 'property uchar red\n' + 'property uchar green\n' + 'property uchar blue\n' + 'end_header\n')

                for index in range(len(output[0]) - 1):
                    z_color_r = 0
                    z_color_r = int(abs((output[2][index] - minus_output2_min) / (minus_output2_max - minus_output2_min)) * 255)
                    file.write(str(output[0][index]) + ' ' + str(output[1][index]) + ' ' + str(output[2][index])+
                               ' 200 ' + str(z_color_r) + ' 0' + '\n')
                    file.flush()
                file.close()
            print("interpolation")
        elif userinput == 2:
            print("Plus Mesh")
        elif userinput == 3:
            print("outlier remove")
        elif userinput == 4:
            print("K key is black background")
            pcd = o3d.io.read_point_cloud(directory + 'result.ply')
            key_to_callback[ord("K")] = change_background_to_black
            o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
        elif userinput == 5:
            manual_registration(directory)
        elif userinput == 0:
            break



print("end")