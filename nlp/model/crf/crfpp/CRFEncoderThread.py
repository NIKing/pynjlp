import threading

class CRFEncoderThread(threading.Thread):
    def __init__(self, wsize):
        self.x = []             # 标记列表
        self.start_i = 0        # 当前线程编号
        self.wSize = 0          # 特征函数长度????
        self.threadNum = 0      # 线程总数量
        self.zeroone = 0
        self.err = 0
        self.size = 0           # 句子长度
        self.obj = 0.0
        self.expected = []

        if wsize > 0:
            self.wSize = wsize
            self.expected = [0.0] * wsize

    def run(self):
        """线程的计算在这里"""
        obj, erro, zeroone = 0.0, 0, 0

        if len(self.expected) == 0:
            self.expected = [0.0]
            

        for i in range(self.start_i, self.size):
            obj += self.x[i].gradient(expected)
            
            errorNum = self.x[i].eval()
            err += errorNum

            if errorNum != 0:
                zeroone += 1

            self.x[i].clearNodes()


        return err


