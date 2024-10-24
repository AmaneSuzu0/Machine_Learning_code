from math import sqrt
from collections import namedtuple


class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # 用来记录当前节点的具体向量
        self.split = split  # 用来记录当前节点的划分维度
        self.left = left  # 左子树
        self.right = right  # 右子树


class KdTree:
    def __init__(self, data):
        k = len(data[0])  # 记录向量(数据)的维度

        def build_tree(split, data_set):  # 递归函数，用来构建KD树
            if not data_set:  # 如果数据集为空，则返回None
                return None

            data_set.sort(key=lambda x: x[split])  # 对数据集按照split维度进行排序,为了选出该维度的中位数作为划分点
            split_pos = len(data_set) // 2  # //作为正数除法来划分数据集
            median = data_set[split_pos]  # 得到split维度下中位数的该数据的具体节点向量
            next_split = (split + 1) % k  # 得到下一个划分维度
            return KdNode(  # 开始构建当前节点
                median,  # 并且递归地构建该节点的左子树和右子树
                split,
                build_tree(next_split, data_set[:split_pos]),
                build_tree(next_split, data_set[split_pos + 1:])
            )

        self.root = build_tree(0, data)


def preorder(node):
    print(node.dom_elt)
    if node.left:
        preorder(node.left)
    if node.right:
        preorder(node.right)


result = namedtuple("Result",  # 定义一个namedtuple，用来保存查询结果
                    ['nearest_point', 'nearest_dist', 'node_visited'])


def find_knn(tree, point):
    k = len(point)  # 记录查询点的维度

    def travel(kd_node, target, max_dist):  # 递归函数，用来查询KD树, target是查询点
        if kd_node is None:  # 如果当前节点为空
            return result([0] * k, float('inf'), 0)

        node_visited = 1  # 记录访问过的节点数
        split_dim = kd_node.split  # 记录当前节点的划分维度
        pivot = kd_node.dom_elt  # 当前节点的具体向量值
        if target[split_dim] < pivot[split_dim]:  # 如果查询点的split维度的值小于等于当前节点的split维度的值
            nearest_node = kd_node.left
            further_node = kd_node.right
        else:
            nearest_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearest_node, target, max_dist)  # 按照查询点的split维度的值，递归地查询当前节点的左子树
        nearest = temp1.nearest_point
        dist = temp1.nearest_dist  # 如果temp1是空，则dist为inf
        node_visited += temp1.node_visited

        if dist < max_dist:  # 如果查询点的距离小于最大距离(首次dist=inf同样max_dist=inf,所以不触发该语句)
            max_dist = dist

        temp_dist = abs(pivot[split_dim] - target[split_dim])  # 计算目标节点距离当前节点划分的超平面的距离
        if max_dist < temp_dist:  # 判断以当前最近节点到目标节点距离构成的超平面是否与本节点(父节点)构成的超平面相交(首次max_dist赋值为inf的时候肯定不走这个if语句)
            return result(nearest, dist, node_visited)  # 不相交,所以继续返回temp就行(回到父节点)

        # ----------------------------------------------------------------走到这说明超平面可能和本节点相交(或者首次max_dist=inf)，需要继续递归查询
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))  # 计算目标节点距离当前节点的欧氏距离(首次当前节点就是叶子节点)
        if temp_dist < dist:  # 如果目标节点距离本节点更近(首次dist=inf的时候肯定走这个if语句)
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        # 再去查询另一个子节点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)  # 如果当前节点是叶子节点(最近节点)的父节点,那么该语句需要进入该节点的另一个叶子节点
        node_visited += temp2.node_visited

        if temp2.nearest_dist < dist:  # 如果另一个子节点的最近点距离本节点更近
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        return result(nearest, dist, node_visited)

    return travel(tree.root, point, float('inf'))


data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
preorder(kd.root)

test_point = [3, 4.5]
result = find_knn(kd, test_point)
print(result)