import numpy as np

class LeastSqTree:
    def __init__(self, train_data, train_label, epsilon):
        self.train_data = train_data
        self.train_label = train_label
        self.epsilon = epsilon
        self.feature_num = train_data.shape[1]
        self.tree = None

    def _fit(self, x, y, feature_num, epsilon):

        f_dim, f_pos, cost, c1, c2 = self._divie(x, y, feature_num)
        tree = {
            'feature': f_dim,
            'value': x[f_pos, f_dim],
            'left': None,
            'right': None,
            'leaf': False,
        }

        if cost < self.epsilon or len(y[np.where(x[:, f_dim] <= x[f_pos, f_dim])]) <= 1:
            tree['left'] = c1
            tree['leaf'] = True
        else:
            tree['left'] = self._fit(
                x[np.where(x[:, f_dim] <= x[f_pos, f_dim])],
                y[np.where(x[:, f_dim] <= x[f_pos, f_dim])],
                self.feature_num,
                self.epsilon,
            )

        if cost < self.epsilon or len(y[np.where(x[:, f_dim] > x[f_pos, f_dim])]) <= 1:
            tree['right'] = c2
            tree['leaf'] = True
        else:
            tree['right'] = self._fit(
                x[np.where(x[:, f_dim] > x[f_pos, f_dim])],
                y[np.where(x[:, f_dim] > x[f_pos, f_dim])],
                self.feature_num,
                self.epsilon,
            )
        return tree

    @staticmethod
    def _divie(x, y, feature_num):
        # 初始化损失误差，[
        # [11],[12],[13]    用来记录用每个Xi划分得到的最小平方误差
        # [21],[22],[23]
        # ]
        cost = np.zeros((feature_num, len(x)))
        for i in range(feature_num):
            for j in range(len(x)):
                value = x[j, i]
                y1 = y[np.where(x[:, i] <= value)]
                c1 = np.mean(y1)
                y2 = y[np.where(x[:, i] > value)]
                c2 = np.mean(y2)
                y1 = (y1 - c1) ** 2
                y2 = (y2 - c2) ** 2
                cost[i, j] = np.sum(y1) + np.sum(y2)

        # 找出最小损失误差,得到其特征维度和划分点
        cost_index = np.where(cost == np.min(cost))  # 返回一个元组，第一个元素是个list行索引，第二个元素是列索引
        f_dim = cost_index[0][0]
        f_pos = cost_index[1][0]
        # 得到了划分点之后，求出按照划分点划分的区域的均值，用来做叶子节点的预测值
        c1 = np.mean(y[np.where(x[:, f_dim] <= x[f_pos, f_dim])])
        c2 = np.mean(y[np.where(x[:, f_dim] > x[f_pos, f_dim])])
        return f_dim, f_pos, cost[f_dim, f_pos], c1, c2

    def fit(self):
        self.tree = self._fit(self.train_data, self.train_label, self.feature_num, self.epsilon)

    def _predict(self, x, tree):
        if tree['leaf']:
            if x[0][tree['feature']] <= tree['value']:
                return tree['left']
            else:
                return tree['right']

        if x[0][tree['feature']] <= tree['value']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

    def predict(self,test_data):
        return self._predict(test_data, self.tree)


train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
model_tree = LeastSqTree(train_X, y, .2)
model_tree.fit()
print(model_tree.predict(np.array([[7.4]])))
