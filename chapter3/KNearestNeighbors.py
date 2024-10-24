import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


def distance(x, y, p=2):
    # x是一个点的坐标，y是另一个点的坐标，p是距离度量的指数
    if len(x) == len(y) and len(x) > 0:  # 两个点的维度必须相同，并且至少有一个维度
        sum_dist = 0  # 距离的和
        for i in range(len(x)):
            sum_dist += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum_dist, 1 / p)
    else:
        return None


# x1, x2, x3 = [1, 1], [5, 1], [4, 4]
# for i in range(1, 5):  # 让p从1到4遍历
#     r = {'p={}, x1-{}'.format(i, c): distance(x1, c, i) for c in [x2, x3]}
#     for j in zip(r.values(), r.keys()):
#         print(j)
# print(r)
# print('最小距离和对应的点：{}'.format(min(zip(r.values(), r.keys()))))

class KNN:
    def __init__(self, x_train, y_train, n_neighbors=3, p=2):
        self.x_train = x_train
        self.y_train = y_train
        self.n = n_neighbors
        self.p = p

    def predict(self, x_test):
        knn_dist = []
        for i in range(len(self.x_train)):
            dist = distance(x_test, self.x_train[i], self.p)
            if i < self.n:  # 先加训练集中前n个点,包括距离和标签
                knn_dist.append((dist, self.y_train[i]))
            else:
                max_index = knn_dist.index(max(knn_dist, key=lambda x: x[0]))  # 找到距离最大的点的索引
                if dist < knn_dist[max_index][0]:  # 如果有比当前的最近的n个节点中最远的节点更近的节点，则替换
                    knn_dist[max_index] = (dist, self.y_train[i])
        knn_y = [j[-1] for j in knn_dist]
        knn_ycount = Counter(knn_y)  # 统计knn_dist中的标签y的数量
        max_ycount = sorted(knn_ycount.items(), key=lambda x: x[1])[-1][0]
        return max_ycount, knn_dist

    def score(self, x_test, y_test):  # 计算预测的准确率
        right_count = 0
        for i in range(len(x_test)):
            if self.predict(x_test[i]) == y_test[i]:
                right_count += 1
        return right_count / len(x_test)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
data = np.array(df.iloc[:100, [0, 1, -1]])  # 取前100条数据, 前两列为特征，最后一列为标签
X, Y = data[:, :-1], data[:, -1]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)  # 划分训练集和测试集

xx_train = np.array([[2, 3],
                     [5, 4],
                     [9, 6],
                     [4, 7],
                     [8, 1],
                     [7, 2]])
yy_train = np.array([0, 0, 0, 0, 1, 1])
test_point = [3, 4.5]
k = 3
clf = KNN(xx_train, yy_train, n_neighbors=k, p=2)
max_y, knn_dist = clf.predict(test_point)

text = ''
for i in knn_dist:
    text += "%.3f" % i[0]+" "

print('最近的{}个点的距离分别是{}'.format(k, text))


# plt.scatter(df.iloc[0:50, 0], df.iloc[0:50, 1], label='0')
# plt.scatter(df.iloc[50:100, 0], df.iloc[50:100, 1], label='1')
# plt.plot(test_point[0], test_point[1], 'go', label='Test Point')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.legend()
# plt.show()
