# 基础知识

- 特征值特征向量
- SVD奇异值分解
- [主成分分析](https://zhuanlan.zhihu.com/p/77151308)、KernelPCA
- 滤波（Radius Outlier Removal、Statistical Outlier Removal）
- 降采样（Voxel Grid Downsampling、Farthest Point Sampling、Normal Space Sampling、S-NET）
- 上采样（Bilateral Filter）

# [lesson1](./lesson1)
- 主要做了 PCA
- Voxel Grid DownSampling

# [lesson2](./lesson2)
- BST
    - 二叉树插入
    - 中序、前序、后序遍历
    - 递归查找、循环查找
    - 1NN 最临近搜索
    - kNN 最邻近搜索
    - radius NN 最邻近搜索
  
- KDTree （二维）
    - KD树构建
    - kNN 最邻近搜索
    - radius 最邻近搜索
  
- OcTree（三维）
    - OcTree树构建
    - kNN 最邻近搜索
    - radius 最邻近搜索
    - radius faster 最邻近搜索
  
# [lesson3](./lesson3))

## [KMeans](./lesson3/KMeans.py)
1. 随机选取 K 个中心点
2. 计算每个点都属于哪一个类
3. 重新计算各个中心点的位置
4. 迭代 2 3 步
  

### [KMeans++](./lesson3/KMeans_kpp.py)
目的： 使得距离所有簇中心越远的点被选中的概率越大，离得越近被随机到的概率越小。

1. 从数据集中随机选取一个点作为初始聚类中心c1
2. 计算每个样本与当前已有聚类中心之间的最短距离（即与最近的一个聚类中心的距离），用D(x)表示；
    接着计算每个样本被选为下一个聚类中心的概率。
    最后，按照**轮盘法**选择出下一个聚类中心
3. 重复第2步直到选出共K个聚类中心；
         之后与经典KMeans中2 3 4 步相同
   
#### [轮盘法](./lesson3/RWS.py)：

根据权重来确定概率：权重越大，被选择的几率越高
这个算法应该源于抽奖，原理也非常相似。

参考:
- https://zhuanlan.zhihu.com/p/140418005
- https://bbs.huaweicloud.com/blogs/detail/155736


# GMM
1. 初始化均值 