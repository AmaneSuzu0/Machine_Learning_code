from math import exp, log
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data(): # 初始化iris数据集，仅取前两列数据作为（x1，x2）和最后一列数据作为标签y
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]


class LogisticRegressionClassifier:
    # 设置最大迭代次数和学习率
    def __init__(self, max_iter=300, learning_rate=0.01):
        self.weight = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    # sigmoid函数
    def sigmoid(self, x):
        return 1/(1 + np.exp(-np.dot(x, self.weight)))

    # p93页对数似然函数L(w)
    def log_fun(self, x, y):
        return np.dot(np.dot(x, self.weight).T, y) - np.sum(np.log(1 + np.exp(np.dot(x, self.weight))))

    # 遍历数据集中的每个数据x，在尾部添1把x变为(x1,x2,.....,xn,1)
    def data_matrix(self, data, label):
        data_mat = []
        for x in data:
            data_mat.append([*x, 1])
        data_mat = np.array(data_mat)
        label = label.reshape(len(label), 1)
        print(data_mat.shape)
        return data_mat, label

    def fit(self, x_train, y_train):
        x_train, y_train = self.data_matrix(x_train, y_train)
        self.weight = np.zeros((len(x_train[0]), 1), dtype=np.float32)  # 将权重w设置为默认的全0
        diff = 1  # 设置两次迭代之间的对数似然函数L(w)的差
        iter_count = 0  # 统计迭代次数
        log_before = self.log_fun(x_train, y_train)  # 当前的对数函数值
        # 如果两次迭代直接似然函数的差非常小，或者到达了最大迭代次数就停止迭代
        while (diff >= 1e-4) and (iter_count < self.max_iter):
            self.weight += self.learning_rate * np.dot(x_train.T, y_train - self.sigmoid(x_train))
            log_after = self.log_fun(x_train, y_train)
            diff = abs(log_after - log_before)
            log_before = log_after
            iter_count += 1
            print("当前迭代次数为{}，两次迭代之间的对数似然函数的差为{}".format(iter_count, diff))

    def score(self, x_test, y_test):
        right = 0.0
        x_test, y_test = self.data_matrix(x_test, y_test)
        for x, y in zip(x_test, y_test):
            result = self.sigmoid(x)
            if (result >= 0.5 and y == 1) or (result < 0.5 and y == 0):
                right += 1
        return right/len(x_test)

    def predict(self, x_test):
        result = self.sigmoid(x_test)
        print("该测试数据类别为1的概率为：{:.3f}%".format(result[0]*100))
        return result


if __name__ == '__main__':
    x, y = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    lr_clf = LogisticRegressionClassifier(max_iter=300)
    lr_clf.fit(x_train, y_train)
    score = lr_clf.score(x_test, y_test)
    print("预测成功率为：{:.3f}%".format(score*100))
    new_test_data = [7, 3, 1]
    lr_clf.predict(new_test_data)


