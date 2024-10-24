import numpy as np
from sklearn.neighbors import KDTree

train_data = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])

tree = KDTree(train_data, leaf_size=2)
dist, idx = tree.query(np.array([(3., 4.5)]), k=1)

x1 = train_data[idx[0]][0][0]
x2 = train_data[idx[0]][0][1]
print('最近的临点是（{}，{}）'.format(x1, x2))
