# 文件功能： 实现 K-Means++ 算法
# KMeans++ : 目的为了使初始的聚类中心之间相互距离要尽可能的远
# 1. 从数据集中随机选取一个点作为初始聚类中心c1
# 2. 计算每个样本与当前已有聚类中心之间的最短距离（即与最近的一个聚类中心的距离），用D(x)表示；
#    接着计算每个样本被选为下一个聚类中心的概率。
#    最后，按照轮盘发选择出下一个聚类中心
# 3. 重复第2步直到选出共K个聚类中心；
#         之后与经典KMeans中2 3 4 步相同
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

    def euler_distance(self, point1: list, point2: list) -> float:
        """
        计算两点之间的欧拉距离，支持多维
        """
        distance = 0.0
        for a, b in zip(point1, point2):
            distance += math.pow(a - b, 2)
        return math.sqrt(distance)

    def get_closest_dist(self, point, centroids):
        min_dist = math.inf  # 初始设为无穷大
        for i, centroid in enumerate(centroids):
            dist = self.euler_distance(centroid, point)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def kpp_centers(self, data):
        # 1. 从数据集中随机选取一个点作为初始聚类中心c1
        self.centers.append(random.choice(data))
        d = [0 for _ in range(len(data))]
        for _ in range(1, self.k_):
            total = 0.0
            for i, point in enumerate(data):
                d[i] = self.get_closest_dist(point, self.centers)  # 与最近一个聚类中心的距离
                total += d[i]
            total *= random.random()
            for i, di in enumerate(d):  # 轮盘法选出下一个聚类中心；
                total -= di
                if total > 0:
                    continue
                self.centers.append(data[i])
                break

    def fit(self, data):
        # 1. kpp 中心点
        self.kpp_centers(data)
        centers = self.centers  # 从 range(data.shape[0]) 数据中，随机抽取self.k_ 作为一个列表
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
