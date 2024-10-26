from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import pandas as pd
import numpy as np



def create_data(): # 初始化iris数据集，仅取前两列数据作为（x1，x2）和最后一列数据作为标签y
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]

if __name__ == '__main__':
    x, y = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = LogisticRegression(max_iter=200)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(score)
