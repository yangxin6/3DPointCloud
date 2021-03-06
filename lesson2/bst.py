import numpy as np
import math

from lesson2.result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, key, value=-1):
        self.left = None
        self.right = None
        self.key = key
        self.value = value

    def __str__(self):
        return "key: %s, value: %s" % (str(self.key), str(self.value))


# 插入
def insert(root: Node, key, value=-1):
    if root is None:
        root = Node(key, value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            pass
    return root


# 中序遍历
def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)


# 前序遍历
def preorder(root):
    if root is not None:
        print(root)
        inorder(root.left)
        inorder(root.right)


# 后序遍历
def postorder(root):
    if root is not None:
        inorder(root.left)
        inorder(root.right)
        print(root)


# 查找  递归
def search_recursive(root: Node, key):
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search_recursive(root.left, key)
    elif key > root.key:
        return search_recursive(root.right, key)


# 查找  循环
def search_iterative(root: Node, key):
    current_node = root
    while current_node is not None:
        if current_node.key == key:
            return current_node
        elif key < current_node.key:
            current_node = current_node.left
        elif key > current_node.key:
            current_node = current_node.right
    return current_node


# 1NN 最邻近搜索
def search_1nn(root: Node, key, worst_distance=float('inf')):
    if root is None or root.key == key:
        return root
    value = math.fabs(key - root.key)
    worst_distance = value if value < worst_distance else worst_distance
    if key < root.key:
        # 遍历左子树
        if math.fabs(key - root.key) < worst_distance:
            if root.right is None:
                return root
            return search_1nn(root.right, key, worst_distance)
        else:
            if root.left is None:
                return root
            return search_1nn(root.left, key, worst_distance)

    elif key > root.key:
        # 遍历右子树
        if math.fabs(key - root.key) < worst_distance:
            if root.left is None:
                return root
            return search_1nn(root.left, key, worst_distance)
        else:
            if root.right is None:
                return root
            return search_1nn(root.right, key, worst_distance)


# kNN Search
def knn_search(root: Node, result_set: KNNResultSet, key):
    if root is None:
        return False
    result_set.add_point(math.fabs(root.key - key), root.value)
    if key <= root.key:
        # 遍历左子树
        if knn_search(root.left, result_set, key):
            return False
        elif math.fabs(root.key - key) < result_set.worstDist():
            return knn_search(root.right, result_set, key)
    elif key > root.key:
        # 遍历右子树
        if knn_search(root.right, result_set, key):
            return False
        elif math.fabs(root.key - key) < result_set.worstDist():
            return knn_search(root.left, result_set, key)


# radius Search
def radius_search(root: Node, result_set: RadiusNNResultSet, key):
    if root is None:
        return False
    result_set.add_point(math.fabs(root.key - key), root.value)
    if key <= root.key:
        # 遍历左子树
        if radius_search(root.left, result_set, key):
            return False
        elif math.fabs(root.key - key) < result_set.worstDist():
            return radius_search(root.right, result_set, key)
    elif key > root.key:
        # 遍历右子树
        if radius_search(root.right, result_set, key):
            return False
        elif math.fabs(root.key - key) < result_set.worstDist():
            return radius_search(root.left, result_set, key)


def main():
    db_size = 100
    k = 5
    radius = 2.0
    data = np.random.permutation(db_size).tolist()
    # 临近搜索把 2 删掉
    data.remove(11)
    root = None
    for i, point in enumerate(data):
        root = insert(root, point, i)

    # inorder(root)

    # 递归查找
    # value = search_recursive(root, 11)
    # print(value)

    # 循环查找
    # value = search_iterative(root, 11)
    # print(value)

    # 1NN 最邻近搜索
    value = search_1nn(root, 11)
    print(value)

    # kNN最邻近搜索
    knn_result_set = KNNResultSet(capacity=k)
    knn_search(root, knn_result_set, 20)
    print(knn_result_set)
    for i in knn_result_set.dist_index_list:
        print(data[i.index])

    # radius search
    radius_result_set = RadiusNNResultSet(radius=radius)
    radius_search(root, radius_result_set, 20)
    print(radius_result_set)
    for i in radius_result_set.dist_index_list:
        print(data[i.index])

if __name__ == '__main__':
    main()
