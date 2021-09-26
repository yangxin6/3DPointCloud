[源码](https://github.com/fxia22/pointnet.pytorch)

## 环境

| 名称        | 版本                        |
| ----------- | --------------------------- |
| 操作系统    | win10                       |
| CPU         | Intel(R) Core(TM) i7-7700HQ |
| GPU         | GTX1050Ti                   |
| RAM         | 8G                          |
| Python      | 3.6                         |
| Pytorch     | 1.6.0                       |
| torchvision | 0.7.0                       |
| cudatoolkit | 10.2                        |



```bash
# CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```





## ShapeNet

### 分类

| 参数名            | 值    |
| ----------------- | ----- |
| nepoch            | 2     |
| batchSize         | 16    |
| num_points        | 2500  |
| workers           | 4     |
| feature_transform | False |

训练时添加对z轴的随即旋转：data_augmentation=True，测试是关闭旋转：data_augmentation=False

<mark>训练结果</mark>

![shapenet-train](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/shapenet-train.7c9wj06mn3g0.png)

<mark>测试结果</mark>

![shapenet-test](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/shapenet-test.60muvcu1hw00.png) ![shapenet-test-2](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/shapenet-test-2.7326f3uh6vc0.png)



### 分割

| 参数名            | 值    |
| ----------------- | ----- |
| nepoch            | 2     |
| batchSize         | 16    |
| class_choice      | Chair |
| workers           | 4     |
| feature_transform | False |

<mark>训练结果</mark>

![shapenet-seg](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/shapenet-seg.7jbn8ar7ysc0.png)

<mark>测试结果</mark>

<img src="https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/chair.s38ouphewq8.png" alt="chair" style="zoom:67%;" />



## ModelNet

### 分类

| 参数名            | 值    |
| ----------------- | ----- |
| nepoch            | 2     |
| batchSize         | 16    |
| num_points        | 2500  |
| workers           | 4     |
| feature_transform | False |

训练时添加对z轴的随即旋转：data_augmentation=True，测试是关闭旋转：data_augmentation=False

<mark>训练结果</mark>

![shapenet-train](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/shapenet-train.7c9wj06mn3g0.png)

<mark>测试结果</mark>

![modelnet40-test-1](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/modelnet40-test-1.1y2d9jdxawbk.png) ![modelnet40-test-1-2](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/modelnet40-test-1-2.6psx8l8rq500.png)



在第一次训练的权重上再次进行训练

<mark>训练结果</mark>

![modelnet40-train2](https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/modelnet40-train2.6wxwwzqbgts0.png)



<mark>测试结果</mark>

<img src="https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/A.22notmx269s0.png" alt="A" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/yangxin6/img-hosting@master/images/L.5gwmx4aoyec0.png" alt="L" style="zoom:67%;" />



第二次训练发现数据 精度会进一步上升，但是效果和 ShapeNet 的效果相差很大