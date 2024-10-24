import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    return datasets, labels


# 计算信息熵
def entropy(data):
    data = pd.DataFrame(data)
    data_len = len(data)
    last_label = data.iloc[:, -1]  # 取出最后一列,也就是全部的标签
    label_count = Counter(last_label)  # 统计每一个标签的数量
    ent = -sum([(ck / data_len) * log(ck / data_len, 2) for ck in label_count.values()])  # 取出每一个标签的数量ck,计算熵
    return ent


# 计算条件熵
def cond_entropy(data, f_axis):  # data根据feature进行划分
    data_len = len(data)
    features = list(set(data.iloc[:, f_axis]))
    feature_set = {}  # 建立一个字典,用于存储每一个特征的子集
    for f in features:
        feature_set[f] = []  # 初始化字典，让f_axis特征划分出的子集为空列表
    for i in range(data_len):
        feature_set[data.iloc[i, f_axis]].append(data.iloc[i])  # 按照f_axis特征把数据实例加入对应的子集
    cond_ent = 0
    for d in feature_set.values():  # 遍历每一个子集
        d_len = len(d)
        if d_len == 0:
            continue
        cond_ent += (d_len / data_len) * entropy(d)
    return cond_ent


# 计算信息增益
def info_gain(data, f_axis):
    return entropy(data) - cond_entropy(data, f_axis)


def cal_every_feature_info_gain(data):
    info_gain_list = []
    for i in range(data.shape[1] - 1):  # 遍历每一个特征
        info_gain_list.append(info_gain(data, i))
    return info_gain_list


datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)

info_gain_list = cal_every_feature_info_gain(train_data)

for i in range(len(info_gain_list)):
    print('特征{}的信息增益为{:.3f}'.format(labels[i], info_gain_list[i]))
