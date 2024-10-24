# 感知机
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
from sklearn.linear_model import Perceptron  # 导入感知机模型

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

# iris_X表示训练数据, iris_label表示训练标签
iris_X = np.array(df.iloc[:100, :2])
iris_label = np.array([-1 if i == 0 else 1 for i in df.iloc[:100, -1]])


class Model:
    def __init__(self):
        self.w = np.ones(iris_X.shape[1], dtype=np.float32)
        self.b = 0.0
        self.rate = 0.1

    def sign(self, x):
        return 1 if self.w.dot(x) + self.b >= 0 else -1

    # 随机梯度下降法
    def fit(self, x_test, y_label):
        is_flag = True
        while is_flag:
            wrong_count = 0
            for i in range(len(x_test)):  # 遍历训练数据集
                x = x_test[i]
                y = y_label[i]
                if self.sign(x) != y:  # 如果该训练实例带入sign函数的结果与标签不符，则更新参数
                    self.w += self.rate * y * x
                    self.b += self.rate * y
                    wrong_count += 1
            if wrong_count == 0:  # 如果本轮循环中所有的训练参数都没有更新，则停止训练
                is_flag = False
        return '训练结束'

    def predict(self, x_test):
        return [self.sign(i) for i in x_test]


# 手搓版感知机
model = Model()
model.fit(iris_X, iris_label)

# sklearn版感知机
clf = Perceptron(fit_intercept=True, max_iter=1000, tol=None, shuffle=True)
# fit_intercept指是否添加截距项，max_iter最大迭代次数，tol停止条件（两次迭代candidate之间的差值小于tol），shuffle是否打乱训练数据集
clf.fit(iris_X, iris_label)
print(clf.coef_, clf.intercept_) # 打印sklearn感知机模型的参数w和截距项b
# 画出决策边界
x_points = np.linspace(4, 7, 100)
y_points = -(model.w[0]*x_points + model.b)/model.w[1]
plt.plot(x_points, y_points, 'r-', label='decision boundary',)
plt.scatter(df.iloc[:50, 0], df.iloc[:50, 1], label='0')
plt.scatter(df.iloc[50:100, 0], df.iloc[50:100, 1], label='1')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# 预测新数据
print(model.predict([[5.1, 3.5], [6.4, 3.2], [5.6, 2.8]]))
print(model.w, model.b)
