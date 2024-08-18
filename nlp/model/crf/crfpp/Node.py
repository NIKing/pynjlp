import math

"""节点类"""
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
        
        self.fVector = []       # 每个节点对应的特征向量，每个特征都有不同标签的节点
        self.lpath = []
        self.rpath = []
    

    def calcAlpha(self):
        """计算当前节点的左侧所有路径损失值累积""" 

        # 注意，alpha 是循环使用的，也就是累积计算
        self.alpha = 0.0
        for p in self.lpath:
            self.alpha = Node.logsumexp(self.alpha, p.cost + p.lnode.alpha, p == self.lpath[0])

        self.alpha += self.cost

    
    def calcBeta(self):
        """计算当前节点的右侧所有路径损失值累积"""
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
        
        # 重复计算每个父节点与当前节点连接路径期望值
        for p in self.lpath:
            p.calcExpectation(expected, Z, size)

    @staticmethod
    def logsumexp(x, y, flg):
        """计算两个数对数和的近似值，为了确保两个数相乘或相除，能保证结果不会下溢或者上溢"""
        if flg:
            return y

        vmin = min(x, y)
        vmax = max(x, y)
        
        # 如果两个数相差比较大，较小的数在较大的数中影响比较小，可以忽略不计
        if vmax > vmin + Node.MINUS_LOG_EPSILON:
            return vmax
        
        # math.exp(x) 返回自然数的指数值，即 e^x 次幂, 在这里貌似是负数形式？，那么exp(vmin-vmax)结果就是小于1的正数
        # math.log2(x) 返回自然数的对数值，即 解以e为底数多少次幂 = x
        # + 1.0 是为了保证，即使 exp() 的值很低，也可以避免下溢
        return vmax + math.log2(math.exp(vmin - vmax) + 1.0)





