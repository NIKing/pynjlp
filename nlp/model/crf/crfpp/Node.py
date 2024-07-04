import math

class Node():

    LOG2 = 0.69314718055

    MINUS_LOG_EPSILON = 50

    def __init__(self):
        self.clear()
    
    def clear(self):
        self.x = 0
        self.y = 0

        self.alpha = 0.0
        self.beta  = 0.0
        self.cost  = 0.0
        self.besetCost = 0.0

        self.prev = None
        
        self.fVector = []
        self.lpath = []
        self.rpath = []
    

    def calcAlpha(self):
        self.alpha = 0.0
        for p in self.lpath:
            self.alpha = Node.logsumexp(self.alpha, p.cost + p.lnode.alpha, p == self.lpath[0])

        self.alpha += self.cost

    
    def calcBeta(self):
        self.beta = 0.0
        for p in self.rpath:
            self.beta = Node.logsumexp(self.beta, p.cost + p.rnode.beta, p == self.rpath[0])

        self.beta += self.cost

    def calcExpectation(self, expected, Z, size):
        """
        计算期望值
        -param expected 输出期望值
        -param Z 规范化因子
        -param size 标签个数
        """
        c = math.exp(self.alpha + self.beta - self.cost - Z)
        
        i = 0
        while self.fVector[i] != -1:
            idx = self.fVector[i] + self.y
            expected[idx] += c

            i += 1
        
        # 重复计算每个子节点
        for p in self.lpath:
            p.calcExpectation(expected, Z, size)

    @staticmethod
    def logsumexp(x, y, flg):
        if flg:
            return y

        vmin = min(x, y)
        vmax = max(x, y)

        if vmax > vmin + Node.MINUS_LOG_EPSILON:
            return vmax

        return vmax + math.log(math.exp(vmin - vmax) + 1.0)





