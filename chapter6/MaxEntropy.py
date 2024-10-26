import numpy as np
from _collections import defaultdict

class MaxEntropy:
    def __init__(self, max_iter):
        self.train_data = None  # 训练数据
        self.train_label = None  # 训练标签
        self.label_set = None  # 标签y的set
        self.feature_num = None  # 特征数目，即训练数据x有多少列
        self.data_len = None  # 数据集长度
        self.fixy = None  # 一个list，针对数据的每一个特征都有一个字典，里面记录了该特征下具体的特征值和标签(xi,y)出现的次数
        self.fixy_num = 0  # 特征函数fixy的数量
        self.w = None  # 特征函数前面的拉格朗日乘子
        self.x_y2idx = None  # 双向索引
        self.idx2x_y = None  # 双向索引
        self.Ep_xy = None  # EP_xy的期望值EP_xy=sum_xy( P_(x,y) * f(x,y) )
        self.max_iter = max_iter  # 最大迭代次数

    def cal_Pwy_x(self, X, y):  # 此处传入的X是一个完整的训练数据(包含所有的特征(x1,x2,x3...,y)
        numerator = 0  # 分子
        Z = 0  # 分母 也就是规范化因子

        for i in range(self.feature_num):  # 求分子
            if (X[i], y) in self.fixy[i]:  # 如果存在(xi, y)的特征函数
                idx = self.x_y2idx[i][(X[i], y)]  # 该idx不光对照了每个特征函数的期望，还对照了特征函数前面的拉格朗日乘子wi
                numerator += self.w[idx]
        numerator = np.exp(numerator)

        for label in self.label_set:  # 求分母
            zi = 0
            for i in range(self.feature_num):
                if (X[i], label) in self.fixy[i]:
                    idx = self.x_y2idx[i][(X[i], label)]
                    zi += self.w[idx]
            Z += np.exp(zi)

        return numerator / Z

    # 计算每个特征函数的经验期望
    def cal_Ep_xy(self):
        Ep_xy = [0] * self.fixy_num  # 一共有多少特征函数就有多少个期望
        for feature in range(self.feature_num):
            # 课本上fixy的期望公式太泛化，实际上fixy特征函数的期望应该是：[fi出现的次数/总数据量]（其实是P_(x,y)）* fi
            for (x, y) in self.fixy[feature]:
                idx = self.x_y2idx[feature][(x, y)]
                # 课本上的特征函数期望遍历的x,y,但是实际上只有（x,y）满足了fi，这个期望值才存在
                Ep_xy[idx] = self.fixy[feature][(x, y)] / self.data_len
        return Ep_xy

    # 计算每个特征函数的期望(非经验期望)
    def cal_Epxy(self):
        Epxy = [0] * self.fixy_num

        for i in range(self.data_len):
            Pwxy = {}
            for label in self.label_set:
                Pwxy[label] = self.cal_Pwy_x(self.train_data[i], label)

            for feature in range(self.feature_num):
                for label in self.label_set:
                    if (self.train_data[i][feature], label) in self.fixy[feature]:
                        idx = self.x_y2idx[feature][(self.train_data[i][feature], label)]
                        Epxy[idx] += Pwxy[label] * (1 / self.data_len)  # P_(x)是取1/N
        return Epxy

    def init_data(self, x_train, y_train):
        self.train_data = x_train
        self.train_label = y_train
        self.data_len = len(y_train)
        self.feature_num = x_train.shape[1]
        self.label_set = set(y_train)

        # fixyDict是一个list里面为每列特征都创建一个字典
        fixyDict = [defaultdict(int) for i in range(self.feature_num)]
        for i in range(self.data_len):
            for j in range(self.feature_num):
                fixyDict[j][(self.train_data[i][j], self.train_label[i])] += 1
        for i in fixyDict:
            self.fixy_num += len(i)
        self.fixy = fixyDict
        self.w = [0] * self.fixy_num

        # 双向索引绑定
        # 为了通过(x,y)能找到对应是第几列的特征，对每个特征创造一个字典
        self.x_y2idx = [{} for i in range(self.feature_num)]
        self.idx2x_y = {}
        index = 0
        for i in range(self.feature_num):  # i就是第几列特征的列索引
            for (x, y) in self.fixy[i]:  # 从大字典中取出第i列特征对应的特征函数fixy,直接对字典遍历得到的(x,y)是key值
                # 双向索引，为每一列特征的每一个特征函数赋予一个index
                self.x_y2idx[i][(x, y)] = index  # 通过第i列的特征函数(x,y)可以找到一个id
                self.idx2x_y[index] = (x, y)  # 通过id也能找到具体的特征函数(x, y)
                index += 1
        # 计算经验期望值
        self.Ep_xy = self.cal_Ep_xy()

    def fit(self, x_train, y_train):
        self.init_data(x_train, y_train)
        for i in range(self.max_iter):
            Epxy = self.cal_Epxy()

            sigmaList = [0] * self.fixy_num  # IIS算法所需要的σ更新变量表

            for j in range(self.fixy_num):
                sigmaList[j] = (1 / self.fixy_num) * np.log(self.Ep_xy[j] / Epxy[j])
            # 参数w的更新
            self.w = [self.w[i] + sigmaList[i] for i in range(self.fixy_num)]
            # 完成单次迭代

        # 训练结束

    def predict(self, x_test):
        result = []
        for label in self.label_set:
            result.append((label, self.cal_Pwy_x(x_test, label)))  # 结果里存放每一个预测的类别和对应的概率
        return result


if __name__ == '__main__':
    # 第一列是标签(代表是否要外出游玩)
    # 第二至第五列分别为特征：天气状态、气温状态、湿度水平、是否有风
    data_set = [['no', 'sunny', 'hot', 'high', 'FALSE'],
                ['no', 'sunny', 'hot', 'high', 'TRUE'],
                ['not sure', 'overcast', 'hot', 'high', 'FALSE'],
                ['yes', 'rainy', 'mild', 'high', 'FALSE'],
                ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
                ['not sure', 'rainy', 'cool', 'normal', 'TRUE'],
                ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
                ['no', 'sunny', 'mild', 'high', 'FALSE'],
                ['not sure', 'sunny', 'cool', 'normal', 'FALSE'],
                ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
                ['not sure', 'sunny', 'mild', 'normal', 'TRUE'],
                ['yes', 'overcast', 'mild', 'high', 'TRUE'],
                ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
                ['no', 'rainy', 'mild', 'high', 'TRUE']]
    data_set = np.array(data_set)
    x_train, y_train = data_set[:, 1:], data_set[:, 0]
    clf = MaxEntropy(max_iter=200)
    clf.fit(x_train, y_train)
    predict_1 = clf.predict(['sunny', 'hot', 'high', 'FALSE'])
    predict_2 = clf.predict(['sunny', 'hot', 'high', 'TRUE'])
    predict_3 = clf.predict(['overcast', 'hot', 'high', 'FALSE'])
    print(predict_1)
    print(predict_2)
    print(predict_3)
