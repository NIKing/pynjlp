import re
from enum import Enum

from nlp.model.crf.crfpp.Node import Node

class QueueElement():
    node = None
    next_ = None

    fx = None
    gx = None

class Mode(Enum):
    TEST = 0
    LEARN = 1

class ReadStatus:
    SUCCESS = 0
    EOF = 1
    ERROR = 2

class TaggerImpl():

    Mode = Mode
    
    ReadStatus = ReadStatus

    def __init__(self, mode):
        self.mode = mode
        self.vlevel = 0
        self.nbest  = 0
        self.ysize  = 0

        self.Z = 0.0

        self.feature_id = 0
        self.thread_id  = 0
        self.lastError  = None
        self.feature_index = None

        self.x = []
        self.node = []      #注意，node 里面的元素数 list
        self.answer = []
        self.result = []

        self.agenda = None
        self.penalty = []
        self.featureCache = []


    def clearNodes(self):
        if not self.node or len(self.node) <= 0:
            return
        
        # 注意，node 里面的元素数 list
        for n in self.node:
            for i in range(len(n)):
                if n[i]:
                    n[i].clear()
                    n[i] = None


    def open(self, featureIndex):
        self.mode = Mode.LEARN
        self.feature_index = featureIndex
        
        # 获取标签集合的长度
        self.ysize = self.feature_index.ysize()
        
        return True

    def read(self, br):
        self.clear()

        status = ReadStatus.SUCCESS
        while True:
            line = br.readline().rstrip('\n')
            
            if line == '':
                return ReadStatus.EOF
            
            if len(line) == 0 or line == '\t':
                break
            
            if not self.add_cols(line):
                print(f'fail to add line: {line}，len: {len(line)}')
                return ReadStatus.ERROR

        return status
    
    def add_cols(self, line):
        cols = re.split("[\t ]", line)
        return self.add(cols)

    def add(self, cols):
        xsize = self.feature_index.getXsize()

        # 训练数据的 gram长度 小于 之前设定的值就报错, 当在训练的时候，需要+1表示添加标签
        if (self.mode == Mode.LEARN and len(cols) < xsize + 1) \
                or (self.mode == Mode.TEST and len(cols) < xsize):
                    print(f'# x is small: size={len(cols)} xsize={xsize}')
                    return False
        
        # 收集语料数据
        self.x.append(cols)
        self.result.append(0)

        tmpAnswer = 0
        
        if self.mode == Mode.LEARN:
            r = self.ysize
            
            # 找到当前单词的标签在集合中的位置
            for i in range(self.ysize):
                if cols[xsize] == self.yname(i):
                    r = i
            
            # 如果找到的值等于默认值，就认为失败，因为默认值为集合长度，而集合的下标不可能超出长度
            if r == self.ysize:
                print('canot find answer')
                return False

            tmpAnswer = r

        self.answer.append(tmpAnswer)

        l = [Node()] * self.ysize
        self.node.append(l)

        return True

    def empty(self):
        #print('xsize', len(self.x))
        return len(self.x) <= 0

    def shrink(self):
        """收缩, 在这里是指对特征数据的瘦身, 实际是实在编译特征向量"""
        if not self.feature_index.buildFeatures(self):
            print('build features failed')
            return False
        
        return True

    def setThread_id(self, thread_id):
        self.thread_id = thread_id

    def getThread_id(self):
        return self.thread_id

    def setFeature_id(self, feature_id):
        self.feature_id = feature_id

    def getFeature_id(self):
        return self.feature_id

    def y(self, i):
        return self.result[i]

    def x_list(self, i):
        return self.x[i]

    def x_str(self, i, j):
        return self.x[i][j]

    def yname(self, i):
        return self.feature_index.getY()[i]

    def size(self):
        return len(self.x)

    def xsize(self):
        return self.feature_index.getXsize()

    def getFeatureCache(self):
        return self.featureCache

    def get_node(self, i, j):
        return self.node[i][j]

    def set_node(self, n, i, j):
        self.node[i][j] = n

    def clear(self):
        if self.mode == Mode.TEST:
            self.feature_index.clear()

        self.lastError = None
        
        self.x.clear()
        self.node.clear()
        self.answer.clear()
        self.result.clear()
        self.featureCache.clear()

        self.Z = 0.0
        self.cost = 0.0

        return True
    
    def eval(self) -> int:
        err = 0
        for i in range(len(self.x)):
            if self.answer[i] == self.result[i]:
                err += 1

        return err

    def gradient(self, expected) -> float:
        """
        计算梯度
        -param expected 梯度向量
        return 损失函数
        """
        if len(self.x) <= 0:
            return 0.0

        self.buildLattice()         # 编译篱笆网络
        self.forwardbackward()      # 前向/后项计算

        s = 0.0
        
        # 计算每个节点的期望值，实际上是计算每句话中的每个单词与四个标签集（1:4）的期望值，很像是隐马尔可夫模型中的发射概率
        for i in range(len(self.x)):
            for j in range(self.ysize):
                self.node[i][j].calcExpectation(expected, self.Z, self.ysize)

        for i in range(len(self.x)):
            fvector = self.node[i][self.answer[i]].fVector
            j = 0
            while fvector[j] != -1:
                idx = fvector[j] + self.answer[i]
                expected[idx] -= 1

                j += 1

            s += self.node[i][self.answer[i]].cost
            lpath = self.node[i][self.answer[i]].lpath

            for p in lpath:
                if p.lnode.y == self.answer[p.lnode.x]:
                    k = 0
                    while p.fvector[k] != -1:
                        idx = p.fvector[k] + p.lnode.y * self.ysize + p.rnode.y
                        expected[idx] -= 1

                        k += 1

                    s += p.cost
                    break
            
            self.viterbi()

            return self.Z - s


    def buildLattice(self):
        """编译篱笆网络"""
        
        if len(self.x) <= 0:
            return
        
        # 重新编译特征，在这里给特征节点（Node）赋值属性，之前一直找不到节点计算的数据从哪儿来，现在知道了
        self.feature_index.rebuildFeatures(self)
        
        # 计算 花费（cost）/ 也是损失值的意思
        for i in range(len(self.x)):
            for j in range(self.ysize):

                self.feature_index.calcCost(self.node[i][j])
                lpath = self.node[i][j].lpath

                for p in lpath:
                    self.feature_index.calcCost(p)

        
        # 增加多倍分解的惩罚
        if len(self.penalty) <= 0:
            return

        for i in range(len(self.x)):
            for j in range(self.ysize):
                self.node[i][j].cost += self.penalty[i][j]


    def forwardbackward(self):
        """前向后向算法"""

        if len(self.x) <= 0:
            return

        for i in range(len(self.x)):
            for j in range(self.ysize):
                self.node[i][j].calcAlpha()

        for i in range(len(self.x) - 1, -1, -1):
            for j in range(self.ysize):
                self.node[i][j].calcBeta()


        self.Z = 0.0
        for j in range(self.ysize):
            self.Z = Node.logsumexp(self.Z, self.node[0][j].beta, j == 0)

    
    def viterbi(self):
        """维特比解码"""
        # 前向计算
        for i in range(len(self.x)):
            for j in range(self.ysize):
            
                bestc = -1e37
                best = None

                lpath = self.node[i][j].lpath
                for p in lpath:
                    cost = p.lnode.bestCost + p.cost + self.node[i][j].cost
                    if cost > bestc:
                        bestc = cost
                        best  = p.lnode

                self.node[i][j].prev = best
                self.node[i][j].bestCost = bestc if best else self.node[i][j].cost

        
        # 后向回溯
        bestc = -1e37
        best = None
        s = len(self.x) - 1

        for j in range(self.ysize):
            if bestc < self.node[s][j].bestCost:
                best = self.node[s][j]
                bestc = self.node[s][j].bestCost

        
        n = best
        while n:
            self.result.append((n.x, n.y))
            n = n.prev

        self.cost = -self.node[len(self.x) - 1][self.result[len(self.x) - 1]].bestCost







