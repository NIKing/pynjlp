
class IntArrayList():
    serialVersionUID = 1908530358259070518L

    def __init__(self, capacity = 1024, linearExpandFactor = 10240):
        self.data = [0] * capacity
        self.size = 0

        self.linearExpandFactor = linearExpandFactor    # 线性递增
        
        self.exponentialExpanding = False   # 是否指数递增
        self.exponentialExpandFactor = 1.5


    def size(self):
        return self.size

    def getLinearExpandFactor(self):
        return self.linearExpandFactor

    def setLinearExpandFactor(self, linearExpandFactor):
        self.linearExpandFactor = linearExpandFactor
    
    def isExponentialExpanding(self):
        """返回是否指数增长"""
        return self.exponentialExpanding

    def setExponentialExpanding(self, multiplyExpanding):
        """设置是否指数增长"""
        self.exponentialExpanding = multiplyExpanding
    
    def getExponentialExpandFactor(self):
        """获取指数增长数据"""
        return self.exponentialExpandFactor

    def setExponentialExpandFactor(self, multiplyExpandFactor):
        """设置指数增长数据"""
        self.exponentialExpandFactor = multiplyExpandFactor

    def set(self, index, value):
        self.data[index] = value

    def get(self, index):
        return self.data[index]

    def removeLast(self):
        self.size -= 1

    def setLast(self, value):
        self.data[self.size - 1] = value

    def getLast(self):
        return self.data[self.size - 1]

    def pop(self):
        self.size -= 1
        return self.data[self.size]


    def append(self, element):
        """在数组尾部追加一个元素"""
        if self.size == len(self.data):
            self.expand()

        self.data[self.size] = element
        self.size += 1

    def expand(self):
        if not self.exponentialExpanding:
            self.data = [0] * (len(self.data) + self.linearExpandFactor)
        else:
            self.data = [0] * (len(self.data) * self.exponentialExpandFactor)

    def loseWeight(self):
        """去掉多余的buffer"""
        if self.size == len(self.data):
            return

        self.data = [0] * self.size



