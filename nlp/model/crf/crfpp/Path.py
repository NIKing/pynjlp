import math

"""边"""
class Path():
    def __init__(self):
        self.clear()

    def clear(self):
        self.rnode = None
        self.lnode = None

        self.fvector = []
        self.cost = 0.0
    
    def add(self, lnode, rnode):
        """"""
        self.lnode = lnode
        self.rnode = rnode
        
        # 注意，这个特别重要!!! 
        # 在把节点赋值给路径的同时，也把路径添加到节点的两边
        lnode.rpath.append(self)
        rnode.lpath.append(self)

    def calcExpectation(self, expected, Z, size):
        """
        计算边的期望值
        -param expected 期望值采集器，其大小与Alpha大小一致
        -param Z 规范化因子
        -param size 标签个数
        """

        # 计算当前边的（左节点权重 + 当前边的损失值 + 右节点权重 - 归一化因子) 损失值
        # 再进行求导计算
        c = math.exp(self.lnode.alpha + self.cost + self.rnode.beta - Z)
        
        i = 0
        while self.fvector[i] != -1:
            idx = self.fvector[i] + self.lnode.y * size + self.rnode.y
            expected[idx] += c

            i += 1
            
