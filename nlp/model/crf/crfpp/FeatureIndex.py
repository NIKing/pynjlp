import math
from abc import ABC, abstractmethod

class FeatureIndex(ABC):

    BOS = ["_B-1", "_B-2", "_B-3", "_B-4", "_B-5", "_B-6", "_B-7", "_B-8"]

    EOS = ["_B+1", "_B+2", "_B+3", "_B+4", "_B+5", "_B+6", "_B+7", "_B+8"]

    def __init__(self):
        self.maxid = 0
        self.alpha = None
        self.alphaFloat = None

        self.threadNum = 1
        self.max_xsize = 0
        self.costFactor = 1.0
        
        self.xsize = 0
        
        self.unigramTempls = []
        self.bigramTempls  = []
        self.y = []                 # 标签集合

        self.templs = ""
    
    def getAlpha(self):
        return self.alpha

    def setAlpha(self, alpha):
        self.alpha = alpha
    
    def ysize(self):
        return len(self.y)

    def size(slef):
        return self.getMaxid()

    def getMaxid(self):
        return self.maxid

    def setMaxid(self, maxid):
        self.maxid = maxid

    def getXsize(self):
        return self.xsize

    def setXsize(self, xsize):
        self.xsize = xsize

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def makeTempls(self, unigramTempls, bigramTempls):
        """
        制作模版, 相当于合并了一元和二元语法模型
        -param unigramTempls 一元模型模版
        -param bigrameTempls 二元模型模版
        return string
        """
        sb = []
        for temp in unigramTempls:
            sb.append(temp)
            sb.append("\n")

        for temp in bigramTempls:
            sb.append(temp)
            sb.append("\n")

        return "".join(sb)

    
    def buildFeatures(self, tagger):
        featureCache = tagger.getFeatureCache()
        tagger.setFeature_id(len(featureCache))
        
        feature = []

        # tagger.size() 语料数量
        for cur in range(tagger.size()):
            if not self.buildFeatureFromTempl(feature, self.unigramTempls, cur, tagger):
                return False

            feature.append(-1)
            featureCache.append(feature)    # 这样会把feature添加到tagger对象中的featureCache吗？

            feature = []

        for cur in range(1, tagger.size()):
            if not self.buildFeatureFromTempl(feature, self.bigramTempls, cur, tagger):
                return False

            feature.append(-1)
            featureCache.append(feature)
            
            feature = []

        return True


    def buildFeatureFromTempl(self, feature, templs, curPos, tagger):
        """
        根据特征模版进行编译特征
        -param feature 特征空间
        -param templs 特征模版，一元语法模版和二元语法模版
        -param curPos tagger实例中语料库的下标
        -param tagger 标记对象，TaggerImpl 实例
        """
        for templ in templs:
            featureID = self.applyRule(templ, curPos, tagger)
            if featureID == None or len(featureID) <= 0:
                print('format error')
                return False
            
            #print('----', templ, curPos, featureID, '++++')
            id = self.getID(featureID)
            if id != -1:
                feature.append(id)

        return True

    def applyRule(self, templ, cur, tagger):
        """
        应用规则
        -param templ 模版特征值, 比如：U3:%x[-2,0]%x[-1,0] 
        -param cur   tagger实例中语料库的下标
        -param tagger 标记对象
        """ 
        sb = []
        for tmp in templ.split("%x"):
            if len(tmp) <= 0:
                continue

            if tmp.startswith('U') or tmp.startswith('B'):
                sb.append(tmp)
                continue

            # 获取模版中定义的索引值
            tuple_ = tmp.split("]")
            idx = tuple_[0].replace('[', '').split(',')
            
            r = self.getIndex(idx, cur, tagger)
            if not r:
                sb.append(r)

            if len(tuple_) > 1:
                sb.append(tuple_[1])
        
        return ''.join(sb)
            
    @abstractmethod
    def getID(self, featureID):
        pass
    
    def getIndex(self, idxStr, cur, tagger):
        """
        获取
        -param idxStr 特征模版中的索引值, 比如：U3:%x[-2,0]%x[-1,0] 中的 [-2,0]
        -param cur tagger实例中语料库的下标
        -param tagger 标记对象
        """
        row, col = int(idxStr[0]), int(idxStr[1])
        pos = row + col

        if row < -len(FeatureIndex.EOS) or row > len(FeatureIndex.EOS) or col < 0 or col >= tagger.xsize():
            return None

        if self.checkMaxXsize:
            max_xsize = max(self.max_xsize, col + 1)
        
        #print('pos=',pos, 'size=', tagger.size())
        if pos < 0:
            return FeatureIndex.BOS[-pos - 1]

        elif pos >= tagger.size():
            return FeatureIndex.EOS[pos - tagger.size()]

        else:
            return tagger.x_str(pos, col)
