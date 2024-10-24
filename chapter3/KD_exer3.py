# k近邻版的kd树
class Node:
    def __init__(self, value, index, left_child, right_child):
        self.value = value.tolist()
        self.index = index
        self.left_child = left_child
        self.right_child = right_child


class KDTree:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.kd_tree = None
        self._create_kdtree(self.data)

    def _split_subtree(self, data, depth=0):
        if data is None or len(data) == 0:
            return None  # 递归终止条件就是分割到data为空
        dim = depth % data.shape[1]  # depth取余数，决定在哪个维度上进行分割
        data = data[np.argsort(data[:, dim])]  # 对data按照dim维度上的大小来排序
        median_index = data.shape[0] // 2
        node_index = [i for i, v in enumerate(self.data) if np.array_equal(v, data[median_index])]  # 找到data中等于data[
        # median_index]的索引
        # 此处不能用if v == data[median_index]，是因为两个nparray比较得到的是一个bool数组，类似[true,false,true]
        return Node(
            value=data[median_index],  # 当前划分节点
            index=node_index,
            left_child=self._split_subtree(data[:median_index], depth + 1),
            right_child=self._split_subtree(data[median_index + 1:], depth + 1)
        )

    def _create_kdtree(self, data):
        self.kd_tree = self._split_subtree(data)

    def __repr__(self):
        return str(self.kd_tree)

    def query(self, data, k=1):
        data = np.asarray(data)
        k_set = self._search(point=data, tree=self.kd_tree, k=k, k_neighbour_set=[])
        dd = [i[0] for i in k_set]
        ii = [i[2] for i in k_set]
        return dd, ii

    @staticmethod
    def _cal_distance(point1, point2):
        # 计算两个点之间的欧氏距离
        return np.sqrt(np.sum((point1 - point2)**2))

    @staticmethod
    def _insert_kneighbour_set(best, tree, distance):
        # 向k近邻集中插入一个节点,并且保持距离最远的节点放在list的最前面
        n = len(best)  # 得到当前k近邻集的大小
        for i, item in enumerate(best):
            if distance > item[0]:  # best中[0]是距离，如果待插入的节点的距离比当前kset中最远的节点的距离还要远，那么插入在该节点前面
                best.insert(i, (distance, tree.value, tree.index))  # 也就是说每次插入新节点，都是从前往后找到第一个kset中距离没有新节点距离远的点，并插入在其前面
                break  # 插入完毕就break
        if len(best) == n:  # 也就是说没有找到地方插入，说明kset中点的距离都比待插入的点的距离更远，那么就直接加到最后
            best.append((distance, tree.value, tree.index))

    def _update_kneighbour_set(self, tree, point, k, best):  # best是个list，里面装着k个最近邻

        # 更新k近邻集
        distance = self._cal_distance(point, tree.value)  # 得到当前的tree节点和目标point之间的距离
        if len(best) == 0:
            best.append((distance, tree.value, tree.index))  # 如果k近邻集为空，则直接加入该点到目标点的距离、坐标、索引三个信息（一个元组）
        elif len(best) < k:  # 如果k近邻集还没满，则直接加入该点到目标点的距离、坐标、索引三个信息（一个元组）
            self._insert_kneighbour_set(best, tree, distance)
        else:
            if best[0][0] > distance:  # 如果k近邻集中最远的点的距离比当前点的距离更远
                best = best[1:]  # 则把最远的点删掉
                self._insert_kneighbour_set(best, tree, distance)  # 然后把当前点插入到k近邻集中
        return best  # 返回更新后的k近邻集

    def _search(self, point, tree=None, k=1, k_neighbour_set=None, depth=0):
        n_dim = len(point)  # 目标点的维度
        # if k_neighbour_set is None:
        #     k_neighbour_set = []  # 如果kset为空，则初始化个空的set
        if tree is None:
            return k_neighbour_set  # 如果树为空，则返回kset,递归终止条件

        if tree.left_child is None and tree.right_child is None:  # 叶子节点，直接更新一下kset
            return self._update_kneighbour_set(tree, point, k, k_neighbour_set)

        # 如果当前节点不是叶子节点
        if point[depth % n_dim] < tree.value[depth % n_dim]:  # 如果目标点在当前节点的左子树中
            direct = 'left'
            next_branch = tree.left_child
        else:
            direct = 'right'
            next_branch = tree.right_child

        if next_branch is not None:  # 如果下一个分支存在
            k_neighbour_set = self._search(point, next_branch, k, k_neighbour_set, depth + 1)  # 递归搜索下一个分支

            # 搜索回来了之后
            temp_dist = abs(tree.value[depth % n_dim] - point[depth % n_dim])  # 计算目标节点到当前节点的n_dim维度为超平面的距离

            if direct == 'left':  # 如果刚从左子树回来
                if k_neighbour_set[0][0] >= temp_dist or len(k_neighbour_set) < k:  # 如果kset中距离最远的点和超平面有交集，或者是kset还没满
                    # 则更新kset
                    k_neighbour_set = self._update_kneighbour_set(tree, point, k, k_neighbour_set)
                    return self._search(point, tree.right_child, k, k_neighbour_set, depth + 1)  # 继续搜索右子树

            else:  # 如果刚从右子树回来
                if k_neighbour_set[0][0] >= temp_dist or len(k_neighbour_set) < k:  # 如果kset中距离最远的点和超平面有交集，或者是kset还没满
                    k_neighbour_set = self._update_kneighbour_set(tree, point, k, k_neighbour_set)
                    return self._search(point, tree.left_child, k, k_neighbour_set, depth + 1)  # 继续搜索左子树

        else:  # 如果准备要进入的分支并不存在，那么只能将当前节点加入kset中
            return self._update_kneighbour_set(tree, point, k, k_neighbour_set)

        return k_neighbour_set  # 如果从next_branch回来了，并且kset中最远的点和超平面也没有交集，就可以继续返回kset了

def print_kneighbour_set(k, ii, dd, data):
    if k == 1:
        text = "x点的最近邻点是"
    else:
        text = "x点的前%d个最近邻点是" % k

    for i, index in enumerate(ii):
        res = data[index]
        if i == 0:
            text += str(res)
        else:
            text += ", "+str(res)
    if k == 1:
        text += "，距离是"
    else:
        text += "，距离分别是"

    for i, dist in enumerate(dd):
        if i == 0:
            text += "%.3f" % dist
        else:
            text += ", %.3f " % dist
    print(text)


if __name__ == '__main__':
    import numpy as np
    X_train = np.array([[2, 3],
                        [5, 4],
                        [9, 6],
                        [4, 7],
                        [8, 1],
                        [7, 2]])

    kdtree = KDTree(X_train)
    k = 3
    dist, ind = kdtree.query([3, 4.5], k=k)
    print(kdtree)
    print_kneighbour_set(k, ind, dist, X_train)
