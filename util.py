from six import iterkeys
from six.moves import range, zip, zip_longest
from collections import defaultdict, Iterable
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Graph(defaultdict):
    """
    以字典的形式存储图信息(也就是邻接表)，其中key是结点的编号，value是相邻结点编号组成的list
    """

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        """返回图中的所有结点"""
        return self.keys()

    def adjacency_iter(self):
        """返回邻接表"""
        return self.items()

    def subgraph(self, nodes={}):
        """给定顶点集合nodes，返回对于的子图"""
        subgraph = Graph()
        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]
        return subgraph

    def check_self_loops(self):
        """检测自循环(也就是某个结点的相邻节点包含自己的情况)"""
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

    def remove_self_loops(self):
        """删除自循环"""
        for x in self:
            if x in self[x]:
                self[x].remove(x)
        return self

    def make_consistent(self):
        """对邻接表中的相邻节点进行排序并去除自循环"""
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        self.remove_self_loops()
        return self

    def make_undirected(self):
        """转换为无向图"""
        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)
        self.make_consistent()
        return self

    def has_edge(self, v1, v2):
        """判断两顶点间是否有边"""
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        """返回给定顶点的度"""
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        return len(self)

    def number_of_edges(self):
        """图中边的数目"""
        return sum([self.degree(x) for x in self.keys()]) / 2  # 所有顶点度的和再除以2

    def number_of_nodes(self):
        """图中顶点的数目"""
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        返回截断随机游走
        path_length:随机游走的长度
        alpha:重新开始的概率
        start:随机游走的起点
        """
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            # 度大于0的点，也就是有相邻节点的点
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    # 以一定的概率重新回到出发顶点
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
    """
    每chunksize个顶点的连接信息为一个chunk
    """
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked

    adjlist = []
    with open(file_) as f:
        for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
    # print(adjlist)
    G = convert_func(adjlist)

    # 转换为无向图
    if undirected:
        G = G.make_undirected()

    return G

def speak():
    print("我会说话")

def parse_adjacencylist_unchecked(f):
    """
    输入：('1 2 3', '2 1','3 1')
    输出：[[1,2,3],[2,1],[3,1]]
    """
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])
    return adjlist

#Graph(<class 'list'>, product a diction key->value
# {35: [887, 1033, 1688, 1956, 8865, 12576, 15670, 18582, 27510, 28290, 28851, 33904, 33907, 35061, 38205, 41714, 44368,
# 45599, 46079, 46431, 48766, 54129, 54131, 56119, 66556, 66563, 66805, 69284, 69296, 78511, 81722, 82098, 82920, 84021,
# 85352, 86359, 97645, 98698, 103482, 103515, 116552, 128540, 132806, 135130, 141342, 141347, 148170, 175291, 178727, 190697,
# 190706, 197054, 198443, 198653, 206371, 210871, 210872, 229635, 231249, 248425, 249421, 254923, 259701, 259702, 263279,
# 263498, 265203, 273152, 286500, 287787, 289779, 289780, 289781, 307015, 335733, 387795, 415693, 427606, 486840, 503883,
# 503893, 513189, 561238, 568857, 573964, 573978, 574009, 574264, 574462, 575077, 575292, 575331, 576725, 576795, 577227,
# 578780, 579008, 592973, 593091, 593105, 593240, 593260, 593813, 594047, 594543, 594649, 594900, 608326, 634902, 634904,
# 634938, 634975, 640617, 646809, 646837, 647408, 647447, 694759, 735303, 787016, 801170, 1050679, 1103960, 1103985, 1109199,
# 1112911, 1113438, 1113831, 1114331, 1117476, 1119505, 1119708, 1120431, 1123756, 1125386, 1127430, 1127913, 1128204, 1128227,
# 1128314, 1128453, 1128945, 1128959, 1128985, 1129018, 1129027, 1129573, 1129683, 1129778, 1130847, 1130856, 1131116,
# 1131360, 1131557, 1131752, 1133196, 1133338, 1136814, 1137466, 1152421, 1152508, 1153065, 1153280, 1153577, 1153853, 1153943,
# 1154176, 1154459], 1033: [35, 1034, 41714, 45605, 1107062], 103482: [35, 27510, 27514, 1114388, 1119140, 1128990], 103515
def from_adjlist_unchecked(adjlist):
    """
    输入：[[1,2,3],[2,1],[3,1]]
    输出：实例化Graph，例如{1:[2,3],2:[1],3:[1]}
    """
    G = Graph()
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors
    return G


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f: # 读取每行的数据
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected: # 无向图，则加相反的边
                G[y].append(x)
    G.make_consistent()
    return G




def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    return walks

