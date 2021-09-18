# 文件功能：实现 GMM 算法

import numpy as np
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from lesson3 import KMeans

plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.k = n_clusters

        self.mu = None  # mean
        self.cov = None  # 协方差矩阵
        self.prior = None  # 先验概率 -> 权重
        self.posteriori = None  # 后验概率       # k*N矩阵


    def fit(self, data):
        # step1 初始化 均值Mu 权重pi 协方差cov
        # init Mu ，使用K-means中心点
        k_means = KMeans.K_Means(n_clusters=self.k)
        k_means.fit(data)
        self.mu = np.asarray(k_means.centers)  # 将mean的初始值为 k-means 的中心点   k*2 矩阵
        self.cov = np.asarray([np.eye(2, 2)] * self.k)  # 初始化的cov为 k*2*2 的单位矩阵 # eye单位矩阵
        self.prior = np.asarray([1 / self.k] * self.k).reshape(self.k, 1)  # 对pi进行均等分  3*1 矩阵
        self.posteriori = np.zeros((self.k, len(data)))  # 后验概率

        for _ in range(self.max_iter):  # 迭代
            # step2 E-step 算出后验概率posteriori --一个点属于哪个类的概率
            for k in range(self.k):
                # 后验概率模型
                # 多元正态分布 multivariate_normal.pdf 多元高斯分布
                self.posteriori[k] = multivariate_normal.pdf(x=data, mean=self.mu[k], cov=self.cov[k])  # 提取每个点的概率密度分布
            # diag 将一维数组元素放在对角线上，方便进行对应的数据乘法运算
            #                     变为3*3对角矩阵 3*3 * 3*N = 3*N
            # ravel 将矩阵里所有元素变为列表
            self.posteriori = np.diag(self.prior.ravel()) @ self.posteriori
            # 归一化
            self.posteriori /= np.sum(self.posteriori, axis=0)  # 后验概率, 3*N矩阵
            # step3 M-step 使用MLE 算出高斯模型三个参数  mu:mean cov：协方差  prior：先验概率
            self.Nk = np.sum(self.posteriori, axis=1)
            self.mu = np.asarray([np.dot(self.posteriori[k], data) / self.Nk[k] for k in
                                  range(self.k)])  # self.posteriori[k]: 3*2  data:n*2  self.Nk[k]:1
            self.cov = np.asarray([np.dot((data - self.mu[k]).T,
                                          np.dot(np.diag(self.posteriori[k].ravel()), data - self.mu[k])) / self.Nk[k]
                                   for k in range(self.k)])  # sel.cov : 3*2*2
            self.prior = np.asarray([self.Nk / self.k]).reshape(self.k, 1)  # self.prior  3*1
        self.fitted = True

    def predict(self, data):
        # 屏蔽开始
        pass
        # 屏蔽结束

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

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化



