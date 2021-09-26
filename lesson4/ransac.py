import math

import numpy as np
import os
import struct
import open3d as o3d
from pandas import DataFrame
from pyntcloud import PyntCloud

from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
from lesson1.voxel_filter import voxel_filter


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    e = 0.6  # e :outline_ratio
    s = data.shape[0]  # 点的数目
    P = 0.99  # 至少得到一个没有异常值的好样本的概率
    N = 1000  # 样本数/迭代次数
    sigma = 0.2  # 数据和模型之间可接受的最大差值
    pre_total = 0  # 上一次inline的点数
    best_a, best_b, best_c, best_d = 0, 0, 0, 0  # 最好模型的参数估计和内点数目,平面表达方程为   aX + bY + cZ +D= 0

    for index in range(N):
        # step1 选择可以估计出模型的最小数据集，对于平面拟合来说，就是三个点
        while True:
            sample_index = random.sample(range(s), 3)  # 重数据集中随机选取3个点
            p = data[sample_index, :]
            if np.linalg.matrix_rank(p) == 3:  # SVD的方法求解矩阵的秩
                break
        # step2 求解模型
        # 先求解法向量
        v1 = p[2] - p[0]  # 向量 p3 -> p1
        v2 = p[1] - p[0]  # 向量 p2 -> p1
        cp = np.cross(v1, v2)  # 向量叉乘求解 平面法向量
        # 求解模型的a,b,c,d
        a, b, c = cp
        d = cp @ p[2]

        # step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
        dist = abs((a * data[:, 0] + b * data[:, 1] + c * data[:, 2] - d) / (np.sqrt(a * a + b * b + c * c)))  # 点到面的距离
        idx_ground = (dist <= sigma)
        total_inliner = np.sum(idx_ground == True)
        if total_inliner > pre_total:
            N = math.log(1 - P) / math.log(1 - pow(total_inliner / s, 3))
            pre_total = total_inliner
            best_a = a
            best_b = b
            best_c = c
            best_d = d
        # 判断是否当前模型已经符合超过 e
        if total_inliner > s * (1 - e):
            break
    print("iters = %f" % N)
    print(best_a, best_b, best_c, best_d)
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]
    # # 屏蔽结束
    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return ground_cloud, segmented_cloud


# 功能：将点云分割为若干份
# 输入：
#     data: 一帧完整点云
# 输出：
#     block_cloud: 分块点云
def split_point(data):
    # 将data进行分割为 4 个 区域：
    # x>=0 y>=0
    x_p = data[:, 0] >= 0
    x_p_y_p = data[x_p][:, 1] >= 0
    x_p_y_p_data = data[x_p][x_p_y_p]

    # x>=0 y<0
    x_p_y_n = data[x_p][:, 1] < 0
    x_p_y_n_data = data[x_p][x_p_y_n]

    # x<0 y<0
    x_n = data[:, 0] < 0
    x_n_y_n = data[x_n][:, 1] < 0
    x_n_y_n_data = data[x_n][x_n_y_n]

    # x<0 y>=0
    x_n_y_p = data[x_n][:, 1] >= 0
    x_n_y_p_data = data[x_n][x_n_y_p]
    return [x_p_y_p_data, x_p_y_n_data, x_n_y_n_data, x_n_y_p_data]


# 功能：从点云文件中分块滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def block_ground_segmentation(data):
    block_points = split_point(data)
    ground_points, segmented_points = ground_segmentation(data=block_points[0])
    for p in block_points[1:]:
        ground_points_, segmented_points_ = ground_segmentation(data=p)
        ground_points = np.r_[ground_points, ground_points_]
        segmented_points = np.r_[segmented_points, segmented_points_]
    return ground_points, segmented_points


def main():
    # 主函数
    origin_points = read_velodyne_bin("../data/data_object_4/007096.bin")
    origin_points_df = DataFrame(origin_points,columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    # 显示原始点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

    # 点云降采样
    # filtered_points = voxel_filter(point_cloud_pynt.points, 0.05, "centroid")
    # filtered_points_df = DataFrame(filtered_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    # filtered_points_pynt = PyntCloud(filtered_points_df)  # 将points的数据 存到结构体中
    # filtered_cloud_o3d = filtered_points_pynt.to_instance("open3d", mesh=False)  # 实例化
    # o3d.visualization.draw_geometries([filtered_cloud_o3d])

    # 地面分割
    ground_points, segmented_points = ground_segmentation(data=origin_points)

    ground_points_df = DataFrame(ground_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_ground = PyntCloud(ground_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_ground = point_cloud_pynt_ground.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])

    segmented_points_df = DataFrame(segmented_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_segmented = PyntCloud(segmented_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_segmented = point_cloud_pynt_segmented.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground, point_cloud_o3d_segmented])

    # 分块的地面分割
    ground_points, segmented_points = block_ground_segmentation(data=segmented_points)
    ground_points_df = DataFrame(ground_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_ground = PyntCloud(ground_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_ground = point_cloud_pynt_ground.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])

    # 显示segmentd_points示地面点云
    segmented_points_df = DataFrame(segmented_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_segmented = PyntCloud(segmented_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_segmented = point_cloud_pynt_segmented.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground, point_cloud_o3d_segmented])


main()
