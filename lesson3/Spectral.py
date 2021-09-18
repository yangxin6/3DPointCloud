# 文件功能：实现 Spectral 谱聚类 算法

import numpy as np
from numpy import *
import scipy
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from lesson2.result_set import KNNResultSet, RadiusNNResultSet
from sklearn.cluster import KMeans
import lesson2.kdtree as kdtree

plt.style.use('seaborn')


def my_distance_Marix(data):
    S = np.zeros((len(data), len(data)))  # 初始化 关系矩阵 w 为 n*n的矩阵
    # step1 建立关系矩阵， 每个节点都有连线，权重为距离的倒数
    for i in range(len(data)):  # i:行
        for j in range(len(data)):  # j:列
            if i == j:
                S[i][j] = 0
            else:
                S[i][j] = np.linalg.norm(data[i] - data[j])  # 二范数计算两个点直接的距离，两个点之间的权重为之间距离的倒数
    return S


def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1 - x2) ** 2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


def kdtree_contribute_Matrix(S, K):
    N = len(S)
    A = np.zeros((N, N))
    leaf_size = 4
    root = kdtree.kdtree_construction(S, leaf_size=leaf_size)
    for i in range(N):
        query = S[i]
        result_set = KNNResultSet(capacity=K)
        kdtree.kdtree_knn_search(root, S, result_set, query)
        index = result_set.knn_output_index()
        for j in index:
            A[i][j] = 1  #
            A[j][i] = A[i][j]
            if i == j:
                A[i][j] = 0
    return A


def myKNN(S, k):
    N = len(S)
    A = np.zeros((N, N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours

        for j in neighbours_id:  # xj is xi's neighbour
            A[i][j] = 1
            A[j][i] = A[i][j]  # mutually
    return A


# 二维点云显示函数
def Point_Show(point, color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y, color=color)


class Spectral(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.k = n_clusters

    def fit(self, data):
        self.W = np.zeros((len(data), len(data)))  # 初始化 关系矩阵 w 为 n*n的矩阵
        # step1 先计算每个点的距离矩阵，再使用KNN建立近邻矩阵
        self.W = kdtree_contribute_Matrix(data, 8)
        # self.W = myKNN(my_distance_Marix(data), 5)
        # step2 换算Laplacian L 拉普拉斯矩阵
        ##换算D矩阵
        self.D = np.diag(np.sum(self.W, axis=1))  # 列相加,并转化为对角线矩阵
        self.L = self.D - self.W  # 拉普拉斯矩阵 L = D - W
        # step3 算拉普拉斯L矩阵最小的K个特征向量记为V
        ###法一
        # eigval, eigvec = np.linalg.eigh(L)
        # features = np.asarray([eigvec[:,i] for i in range(self.__K)]).T
        ###法二
        _, self.Y = scipy.linalg.eigh(self.L, eigvals=(0, 2))  # 特征值分解
        # step4 把 N*k维 向量 进行K-means聚类
        # k_means = KMeans.K_Means(n_clusters=self.k)       #初始化kmeans
        # k_means.fit(self.Y)
        # result = k_means.predict(self.Y)
        sp_kmeans = KMeans(n_clusters=self.k).fit(self.Y)
        self.label = sp_kmeans.labels_
        return sp_kmeans.labels_

    def predict(self, data):
        """
        Get cluster labels
        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray
        Returns
        ----------
        result: numpy.ndarray
            data labels as (N, ) numpy.ndarray
        """
        return np.copy(self.label)


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    # X = np.array([[1, 2], [2, 3], [5, 8], [8, 8], [1, 6], [9, 11]])

    spectral = Spectral(n_clusters=3)
    K = 3
    spectral.fit(X)
    cat = spectral.predict(X)
    print(cat)
    cluster = [[] for i in range(K)]
    for i in range(len(X)):
        if cat[i] == 0:
            cluster[0].append(X[i])
        elif cat[i] == 1:
            cluster[1].append(X[i])
        elif cat[i] == 2:
            cluster[2].append(X[i])
    Point_Show(cluster[0], color="red")
    Point_Show(cluster[1], color="orange")
    Point_Show(cluster[2], color="blue")
    plt.show()
