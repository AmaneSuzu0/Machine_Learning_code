import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, f_value=None, left=None, right=None, leaf=False, label=None):
        self.label = label
        self.feature = feature
        self.left = left
        self.right = right
        self.f_value = f_value  # 在特征值f下用来划分的
        self.leaf = leaf  # 是否是叶子节点

    def __repr__(self):
        return "Node(feature={}, f_value={}, \nleft={}, right={}, \nleaf={}, label={})".format(self.feature, self.f_value, self.left, self.right, self.leaf, self.label)

class CartTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.root = None

    def _fit(self, data):
        data_x, data_y, features = data.iloc[:, :-1], data.iloc[:, -1], data.columns[:-1]
        # 若数据集中所有样本的标签相同，则返回叶子节点
        if len(set(data_y)) == 1:
            label_count = data_y.value_counts()
            label = label_count.idxmax()
            return Node(leaf=True, label=label)
        # 若特征集为空，则返回叶子节点
        if len(features) == 0:
            label_count = data_y.value_counts()
            label = label_count.idxmax()
            return Node(leaf=True, label=label)

        # 计算每个特征的基尼指数
        feature_val_list = []
        for f_axis in range(len(features)):
            f_value, mini_gini = self.get_min_gini(data, f_axis)  # 返回的是f_axis特征下基尼系数最小的value和基尼系数
            feature_val_list.append((mini_gini, f_axis, f_value))

        feature_val_list = sorted(feature_val_list, key=lambda x: x[0])
        best_gini = feature_val_list[0][0]
        best_f_axis = feature_val_list[0][1]
        best_f_value = feature_val_list[0][2]
        # 若基尼指数小于阈值，则返回叶子节点
        if best_gini < self.epsilon:
            label_count = data_y.value_counts()
            label = label_count.idxmax()
            return Node(leaf=True, label=label)
        print("最佳特征：{}，最佳特征值：{}".format(features[best_f_axis], best_f_value))
        d_true = data[data.iloc[:, best_f_axis] == best_f_value].drop(columns=features[best_f_axis])
        d_false = data[data.iloc[:, best_f_axis] != best_f_value].drop(columns=features[best_f_axis])
        left_node = self._fit(d_true)
        right_node = self._fit(d_false)
        return Node(feature=features[best_f_axis], f_value=best_f_value, left=left_node, right=right_node)

    # 计算基尼指数
    def cal_gini(self, data):
        data_len = len(data)
        label_set = list(set(data.iloc[:, -1]))
        label_count = {}
        for label in label_set:
            label_count[label] = 0
        for i in range(data_len):
            label_count[data.iloc[i, -1]] += 1

        return 1 - sum([(i / data_len) ** 2 for i in label_count.values()])

    # 计算条件基尼指数
    def cond_cal_gini(self, data, f_axis, f_value):
        data_len = len(data)
        d_true = data[data.iloc[:, f_axis] == f_value]  # 取出A特征值为f_value的样本
        d_false = data[data.iloc[:, f_axis] != f_value]  # 取出A特征值不为f_value的样本
        true_len = len(d_true)
        false_len = len(d_false)
        return (true_len / data_len) * self.cal_gini(d_true) + (false_len / data_len) * self.cal_gini(d_false)

    def get_min_gini(self, data, f_axis):
        data_len = len(data)
        f_set = list(set(data.iloc[:, f_axis]))
        gini_list = []
        for f_value in f_set:
            gini_list.append((f_value, self.cond_cal_gini(data, f_axis, f_value)))

        gini_list = sorted(gini_list, key=lambda x: x[1])
        print(gini_list)
        return gini_list[0]

    def fit(self, data):
        self.root = self._fit(data)

    def _predict(self, test_data, node):
        if node.leaf:
            return node.label

        if test_data[node.feature] == node.f_value:
            return self._predict(test_data, node.left)
        else:
            return self._predict(test_data, node.right)


    def predict(self, test_data):

        pre_list = []
        for i in range(len(test_data)):
            pre_list.append(self._predict(test_data.iloc[i, :], self.root))
        return pre_list


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
cart = CartTree(epsilon=0)
cart.fit(train_data)
test_data = train_data.iloc[:, :-1]
test_label = train_data.iloc[:, -1]
print("训练集准确率：{:.3f} %".format(sum(cart.predict(test_data) == test_label) / len(test_label) * 100))
# print(cart.root)

# def create_data():
#     iris = load_iris()
#     df = pd.DataFrame(iris.data, columns=iris.feature_names)
#     df['label'] = iris.target
#     df.columns = [
#         'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
#     ]
#     data = df.iloc[:100, [0, 1, -1]]
#     # print(data)
#     return data.iloc[:, :2], data.iloc[:, -1]
#
#
# X, y = create_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# train_data = pd.concat([X_train, y_train], axis=1)
# cart = CartTree(epsilon=0.01)
# cart.fit(train_data)
# test_data = X_test
# test_label = y_test
