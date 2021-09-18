# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import math
import random

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
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


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    from sklearn.cluster import k_means
    kmeans = k_means(data, n_clusters=5)

    clusters_index = kmeans[1]

    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = '../data/data_object_4/' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        _, segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
