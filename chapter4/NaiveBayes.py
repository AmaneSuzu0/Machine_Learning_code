import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target

    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]  # 返回特征和标签


class NaiveBayes:
    def __init__(self):
        self.label_count = {}
        self.model = None

    @staticmethod
    def mean(x):
        return sum(x) / float(len(x))

    def std(self, x):
        avg = self.mean(x)
        return math.sqrt(sum([pow(i - avg, 2) for i in x]) / float(len(x)))

    def gussian_probability(self, x, mean, std):  # 计算高斯分布概率密度函数,传入的x是待计算的点的一个维度的值，mean和std是该维度下的均值和标准差
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
        return (1.0 / (math.sqrt(2 * math.pi) * std)) * exponent

    def summarize(self, train_data):
        summaries = [(self.mean(i), self.std(i)) for i in zip(*train_data)]  # 把训练数据按列分割，计算每列的均值和标准差(形成一个元组)，返回一个列表
        return summaries  # 类似[(mean1,std1),(mean2,std2),(mean3,std3)]

    def fit(self, train_data, train_label):
        labels = list(set(train_label))  # 获得标签的集合
        for label in labels:  # 统计每个标签的数量（先初始化为0）
            self.label_count[label] = 0
        data = {label: [] for label in labels}  # 建立标签到数据的映射
        for f, label in zip(train_data, train_label):  # f就是每行数据，label就是每行数据的标签
            data[label].append(f)  # 将数据按标签分类
            self.label_count[label] += 1  # 统计每个标签的数量
        self.model = {
            label: self.summarize(value) for label, value in data.items()  # 得到每个标签中全部数据的各个维度上的均值和标准差
        }  # 其实model就是一个字典，key是标签，value是该标签下各个维度的均值和标准差
        return 'gaussianNB train done!'

    def cal_probilities(self, input_data):
        probalities = {}
        total_count = sum(v for v in self.label_count.values())  # 计算总数据量

        for label, value in self.model.items():
            probalities[label] = 1 * self.label_count[label] / total_count  # 计算每个标签的概率
            for i in range(len(value)):
                mean, std = value[i]  # 取出第i个维度的均值和标准差
                probalities[label] *= self.gussian_probability(input_data[i], mean, std)  # P(x=input_data[i]|y=ck)的连乘
        return probalities

    def predict(self, input_data):
        sored_probalities = sorted(
            self.cal_probilities(input_data).items(),
            key=lambda x: x[-1]  # 按概率大小排序(probability中存放的是标签及其概率值)
        )  # 取倒数第一个字典的key，即标签

        label = sored_probalities[-1][0]  # 取概率最大的标签
        return label, sored_probalities

    def score(self, x_test, y_test):
        right = 0
        for x, y in zip(x_test, y_test):
            label, probs = self.predict(x)
            if label == y:
                right += 1
        return right / float(len(x_test))


X, Y = create_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
model = NaiveBayes()
model.fit(x_train, y_train)
real_data = [5.1, 3.5, 1.4, 0.2]
real_label = 1.0
x_test = np.vstack((x_test, real_data))
y_test = np.append(y_test, real_label)
label, probalities = model.predict(real_data)
print("真实的数据：{}".format(real_data))
print("真实标签：{}".format(real_label))
print("预测标签：{}".format(label))
print("预测的概率：{}".format(probalities))
print("{}%".format(model.score(x_test, y_test)*100))
print(x_test[-1], y_test[-1])
