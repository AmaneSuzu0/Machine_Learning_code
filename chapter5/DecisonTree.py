from collections import Counter
import math
from math import log
import numpy as np
import pandas as pd


class Node:
    def __init__(self, leaf=True, label=None, feature_name=None, feature=None):
        self.leaf = leaf  # leaf=True代表是叶子节点，Flase代表是内部节点
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.leaf:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    @staticmethod
    def entropy(data):
        data = pd.DataFrame(data)
        data_len = len(data)
        last_label = data.iloc[:, -1]  # 取出最后一列,也就是全部的标签
        label_count = Counter(last_label)  # 统计每一个标签的数量
        ent = -sum([(ck / data_len) * log(ck / data_len, 2) for ck in label_count.values()])  # 取出每一个标签的数量ck,计算熵
        return ent

    # 计算条件熵
    def cond_entropy(self, data, f_axis):  # data根据feature进行划分
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
            cond_ent += (d_len / data_len) * self.entropy(d)
        return cond_ent

    # 计算信息增益/信息增益比
    def info_gain(self, data, f_axis):
        # 信息增益：
        # return self.entropy(data) - self.cond_entropy(data, f_axis)
        # 信息增益比：
        data_len = len(data)
        feature_set = list(set(data.iloc[:, f_axis]))
        feature_count = {}
        for f in feature_set:
            feature_count[f] = 0
        for i in range(data_len):
            feature_count[data.iloc[i, f_axis]] += 1

        had = -sum([(i/data_len)*log(i/data_len, 2) for i in feature_count.values()])
        return (self.entropy(data) - self.cond_entropy(data, f_axis))/had

    def cal_best_feature_info_gain(self, data):
        info_gain_list = []
        for i in range(data.shape[1] - 1):  # 遍历每一个特征
            info_gain_list.append((i, self.info_gain(data, i)))
        best_f_infogain = max(info_gain_list, key=lambda x: x[1])
        return best_f_infogain

    def train(self, train_data):
        # 得到数据集x和标签y 和所有的特征features
        x_train, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]

        # 如果特征为空
        if len(features) == 0:
            return Node(leaf=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 如果标签只有一种
        if len(y_train.value_counts()) == 1:
            return Node(leaf=True, label=y_train.iloc[0])

        # 计算信息增益最大的特征
        max_feature, max_info_gain = self.cal_best_feature_info_gain(train_data)
        max_feature_name = features[max_feature]

        if max_info_gain < self.epsilon:
            return Node(leaf=True, label=y_train.value_counts().sort_values(ascendin=False).index[0])

        # 创建节点，并且按照我们上面挑选出来的特征赋予其feature属性
        node_tree = Node(leaf=False, feature_name=max_feature_name, feature=max_feature)
        # 按照选出来的特征进行划分d1-dn子集
        feature_list = list(set(train_data[max_feature_name]))
        for f in feature_list:  # 遍历我们所选的特征列的每个类别
            # df子集=原数据集中特征列类别为f的几行，再删去特征列
            sub_d = train_data[train_data[max_feature_name] == f].drop(max_feature_name, axis=1)
            # 递归调用train函数，创建子节点
            sub_tree = self.train(sub_d)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, test_data):
        return self._tree.predict(test_data)


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


datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)
dtree = DTree()
dtree.fit(train_data)
print(dtree.predict(['青年', '是', '否', '好']))
