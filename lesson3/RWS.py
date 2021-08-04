# 轮盘算法


import numpy as np


def RWS(P, r):
    """某个个体被选中的概率"""
    q = 0  # 累计概率
    for i in range(1, len(P) + 1):
        q += P[i - 1]  # P[i]表示第i个个体被选中的概率
        if r <= q:  # 产生的随机数在m~m+P[i]间则认为选中了i
            return i


def choice_by_RouletteWheel(P):
    """对所有个体利用轮盘法进行选择"""
    choice_list = {}
    r_list = [0.450126, 0.110347, 0.572496, 0.98503]
    for j in range(len(P)):
        #         r=np.random.random()  #  r为0至1的随机数
        curr_choice = RWS(P, r_list[j])
        if curr_choice in choice_list:
            choice_list[curr_choice] += 1
        else:
            choice_list[curr_choice] = 1
    # 字典按键排序
    result = {}
    for i in sorted(choice_list):  # 对键进行升序排序
        result[i] = choice_list[i]
    return result


if __name__ == '__main__':
    P = [0.14, 0.49, 0.06, 0.31]
    choice_list = choice_by_RouletteWheel(P)
    print("个体被选中次数")
    print(choice_list)
