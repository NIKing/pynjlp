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
        -param expected 输出期望值
        -param Z 规范化因子
        -param size 标签个数
        """
        c = math.exp(self.lnode.alpha + self.cost + self.rnode.beta - Z)
        
        i = 0
        while self.fvector[i] != -1:
            idx = self.fvector[i] + self.lnode.y * size + self.rnode.y
            expected[idx] += c

            i += 1
            
