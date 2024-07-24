import threading

class CRFEncoderThread(threading.Thread):
    def __init__(self, wsize):
        super().__init__()

        self.x = []             # 训练数据中的句子集合，句子是TaggerImpl对象
        
        self.start_i = 0        # 当前线程编号
        self.wSize = 0          # 特征索引对象的最大值
        self.threadNum = 0      # 线程总数量
        self.size = 0           # 训练数据中句子的数量
        
        self.zeroone = 0        # 0|1数量，错误频率
        self.err = 0            # 错误数
        self.obj = 0.0          # 梯度值
        self.expected = []      # 期望值

        if wsize > 0:
            self.wSize = wsize
            self.expected = [0.0] * wsize

    def run(self):
        """线程的计算在这里"""
        self.obj, self.err, self.zeroone = 0.0, 0, 0
        
        # 期望值
        if len(self.expected) == 0:
            self.expected = [0.0] * self.wSize
            
        # 以当前线程为起点，以线程总数为步长，训练所有句子。
        # 比如：线程1训练[1,4,7]句子，线程2训练[2,6,9]这些句子
        for i in range(self.start_i, self.size, self.threadNum):
            # 计算坡度，应该是梯度计算，寻找最低点, 这里调用 TaggerImpl 对象
            self.obj += self.x[i].gradient(self.expected)
           
            # 评测
            errorNum = self.x[i].eval()
            self.err += errorNum

            if errorNum != 0:
                self.zeroone += 1

            self.x[i].clearNodes()

        return self.err


