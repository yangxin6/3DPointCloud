# 文件功能： 实现 K-Means 算法
# 1. 随机选取 K 个中心点
# 2. 计算每个点都属于哪一个类
# 3. 重新计算各个中心点的位置
# 4. 迭代 2 3 步
import math
import random
import numpy as np


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = []
        self.fitted = False

    def fit(self, data):
        # 1. 随机选取 K 个中心点
        centers = data[random.sample(range(data.shape[0]), self.k_)]  # 从 range(data.shape[0]) 数据中，随机抽取self.k_ 作为一个列表
        old_centers = np.copy(centers)  # 将旧的中心点 保存到old_centers
        labels = [[] for _ in range(self.k_)]  # 依据 K 构建空 labels

        for iter_ in range(self.max_iter_):  # 循环一定的次数
            # E Step
            # 2. 计算每个点都属于哪一个类
            for idx, point in enumerate(data):  # enumerate 函数用于一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据的下标
                # 默认的二范数， axis=1 求每行的范数（对应一个点到 K个类对应的点 的距离）
                diff = np.linalg.norm(old_centers - point, axis=1)  # 一个点分别到两个中心点的距离不同
                diff2 = (np.argmin(diff))  # np.argmin(diff) 表示最小值在数组中的位置  选取距离小的那一点的索引 也就代表了属于哪个类
                labels[diff2].append(idx)  # 选取距离小的那一点的索引 也就代表了属于哪个类

            # M Step
            # 3. 重新计算各个中心点的位置
            for i in range(self.k_):
                points = data[labels[i], :]  # 所有在第k类中的所有点
                centers[i] = points.mean(axis=0)  # 均值 作为新的聚类中心
            if np.sum(np.abs(
                    centers - old_centers)) < self.tolerance_ * self.k_:  # 如果前后聚类中心的距离相差小于self.tolerance_ * self.k_ 输出
                break
            old_centers = np.copy(centers)
        self.centers = centers
        self.fitted = True

    def predict(self, p_datas):
        result = []
        if not self.fitted:
            print('Unfitted. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers - point, axis=1)
            result.append(np.argmin(diff))
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
