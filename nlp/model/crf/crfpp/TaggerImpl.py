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

        self.Z = 0

        self.feature_id = 0
        self.thread_id  = 0
        self.lastError  = None
        self.feature_index = None

        self.x = []
        self.node = []
        self.answer = []
        self.result = []

        self.agenda = None
        self.penalty = []
        self.featureCache = []

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
        """收缩, 在这里是指对特征数据的瘦身"""
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



