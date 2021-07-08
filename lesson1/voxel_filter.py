#!/usr/bin/env python
# coding: utf-8

# # 实现voxel滤波，并加载数据集中的文件进行验证

# In[3]:


# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import random
from pyntcloud import PyntCloud
from pandas import DataFrame

def get_all_index(lst, item):
    return [i for i in range(len(lst)) if lst[i] == item]
    
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, mode='centroid'):
    filtered_points = []
    data = point_cloud
    # 作业3
    # 屏蔽开始
    #1.计算点云的最大最小值
    D_min = data.min(0)
    D_max = data.max(0)
    #2.设定划分体素大小，计算空间划分份数1ｘ3
    D = (D_max - D_min) / leaf_size
    #3.每个点计算划分索引
    point_x, point_y, point_z = np.array(data.x), np.array(data.y), np.array(data.z)
    hx = np.floor((point_x - D[0]) / leaf_size)
    hy = np.floor((point_y - D[1]) / leaf_size)
    hz = np.floor((point_z - D[2]) / leaf_size)
    index = np.array(np.floor(hx + hy * D[0] + hz * D[0] * D[1]))   #Nｘ1
    
    #4.对索引进行排序
    data_index_point = np.c_[index, point_x, point_y, point_z]
    sort_idx = data_index_point[:, 0].argsort()
    data_index_point = data_index_point[sort_idx]
    size = data_index_point.shape[0]
    tem_point = []
    
    if mode == 'random':
        #使用随机采样方法,索引相同的点选取最后一个为滤波输出点，相当于是随机采样了
        for i in range(size - 1):
            if(data_index_point[i][0] != data_index_point[i+1][0]):
                filtered_points.append(data_index_point[i][1:])
        #最后一个没有比较，加上
        filtered_points.append(data_index_point[size-1][1:])
        filtered_points = np.array(filtered_points)
    elif mode == 'centroid':
        #使用计算均值方法
        for i in range(size - 1):
            #判断前一个序号和后一个是否相等
            if data_index_point[i][0] == data_index_point[i+1][0]: #对于只有两个点的就会只保留一个点
                tem_point.append(data_index_point[i][1:])
                continue
            if tem_point == []:
                continue
            filtered_points.append(np.mean(tem_point, axis=0))
            tem_point = []
        filtered_points = np.array(filtered_points)
    
    # 屏蔽结束
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


# In[1]:


import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud



# 功能：对点云进行voxel滤波, 使用 hash table 优化排序问题
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter_hash(point_cloud, leaf_size):
    filtered_points = []
    point_cloud = np.asarray(point_cloud)
    # 作业3
    # 屏蔽
    #三个维度最小/大值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
 
    #确定每一个维度的格子数量
    Dx = (x_max - x_min)//leaf_size + 1    #保证0-leaf_size 在第一个格子内
    Dy = (y_max - y_min)//leaf_size + 1
    Dz = (z_max - z_min)//leaf_size + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
 
    dict = { }  #建立一个空的字典 放h对应点的数据
    index_ = { } #放h对应点的数量
    for i in range(len(point_cloud)):
        hx = (point_cloud[i, 0] - x_min)//leaf_size + 1
        hy = (point_cloud[i, 1] - y_min)//leaf_size + 1
        hz = (point_cloud[i, 2] - z_min)//leaf_size + 1
        h = hx + hy*Dx + hz*Dx*Dy
        if (h not in dict):
            dict[h] = point_cloud[i]
            index_[h] = 1
        else:
            val = dict.get(h, 0)  #先把字典中的数据取出来
            num = index_.get(h, 0)
            dict[h] = (val * num + point_cloud[i])/(num + 1) #来一次点就需要求相同h的所有点的平均
            index_[h] = num + 1
 
    for key,value in dict.items():#当两个参数时
        filtered_points.append(value)
    # 屏蔽结束
 
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


# In[8]:


# # 从ModelNet数据集文件夹中自动索引路径，加载点云

# 加载自己的点云文件
point_cloud_raw = np.genfromtxt(r"./car_0005.txt", delimiter=",")  #为 xyz的 N*3矩阵
point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
# o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

# 调用voxel滤波函数，实现滤波
# filtered_cloud = voxel_filter(point_cloud_pynt.points, 20, "random")
filtered_cloud = voxel_filter_hash(point_cloud_pynt.points, 0.05)
point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
# 显示滤波后的点云
o3d.visualization.draw_geometries([point_cloud_o3d])


# In[ ]:




