import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
data = np.array([[5, 12, 1], [6, 21, 0], [14, 5, 0], [16, 10, 0], [13, 19, 0],
                 [13, 32, 1], [17, 27, 1], [18, 24, 1], [20, 20, 0], [23, 14, 1],
                 [23, 25, 1], [23, 31, 1], [26, 8, 0], [30, 17, 1],
                 [30, 26, 1], [34, 8, 0], [34, 19, 1], [37, 28, 1]])

x_train = data[:, :2]
y_train = data[:, 2]

kn1_n, kn2_n = 2, 3  # 两个K值
models = (KNeighborsClassifier(n_neighbors=kn1_n, n_jobs=-1), KNeighborsClassifier(n_neighbors=kn2_n, n_jobs=-1))
models = (clf.fit(x_train, y_train) for clf in models)  # 训练两个模型（但由于是懒惰学习，所以只做了一些准备工作，并非实际的训练）
titles = ('K Neighbors with k={}'.format(kn1_n), 'K Neighbors with k={}'.format(kn2_n))
fig = plt.figure(figsize=(15, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
x0, x1 = x_train[:, 0], x_train[:, 1]

x_min, x_max = x0.min() - 1, x0.max() + 1  # 坐标轴范围
y_min, y_max = x1.min() - 1, x1.max() + 1
# 画网格
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

for clf, title, ax in zip(models, titles, fig.subplots(1, 2).flatten()):  # flatten把两个子图合并为一个列表
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # 把xx和yy展开成一维数组，然后再用np.c_按照两列的形式拼接起来
    z = z.reshape(xx.shape)  # 把z变成和xx一样n*n的方阵
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(z))])
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.4)
    ax.scatter(x0, x1, c=y_train, s=50, edgecolors='k', cmap=cmap)
    ax.set_title(title)
plt.show()



