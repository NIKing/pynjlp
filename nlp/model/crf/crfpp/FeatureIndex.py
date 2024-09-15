import math
from abc import ABC, abstractmethod

from nlp.model.crf.crfpp.Node import Node
from nlp.model.crf.crfpp.Path import Path

class FeatureIndex(ABC):

    BOS = ["_B-1", "_B-2", "_B-3", "_B-4", "_B-5", "_B-6", "_B-7", "_B-8"]

    EOS = ["_B+1", "_B+2", "_B+3", "_B+4", "_B+5", "_B+6", "_B+7", "_B+8"]

    def __init__(self):
        self.maxid = 0
        self.alpha = []
        self.alphaFloat = []

        self.threadNum = 1
        self.max_xsize = 0
        self.costFactor = 1.0
        
        self.xsize = 0
        
        self.unigramTempls = []
        self.bigramTempls  = []
        self.y = []                 # 通过语料库收集标签（去重）

        self.templs = ""
    
    def getAlpha(self):
        return self.alpha

    def setAlpha(self, alpha):
        self.alpha = alpha
    
    def ysize(self):
        return len(self.y)

    def size(self):
        return self.getMaxid()

    def getMaxid(self):
        return self.maxid

    def setMaxid(self, maxid):
        """所有特征的标记总数"""
        self.maxid = maxid

    def getXsize(self):
        return self.xsize

    def setXsize(self, xsize):
        self.xsize = xsize

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y
    
    def test_alpha(self):
        # 看看现在时刻，特征空间的权重有咩用值？
        msg = '没有值'
        for alpha in self.alpha:
            if alpha > 0:
                msg = f'有值={alpha}'
                break

        return msg

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
        """
        编译特征模版
        -param tagger TaggerImpl类 在Encoder代码中读取特征的时候碰到断句会自动创建一个 tagger 对象，因此一个 tagger 可以认为是一个句子的特征对象
        """
        featureCache = tagger.getFeatureCache()
        tagger.setFeature_id(len(featureCache))
        
        feature = []

        # tagger.size() 句子的长度
        # 根据一元语法模版编译句子的一元语法特征
        for cur in range(tagger.size()):

            # 在buildFeatureFromTempl() 中会添加特征索引到 feature 
            if not self.buildFeatureFromTempl(feature, self.unigramTempls, cur, tagger):
                return False

            feature.append(-1)              
            featureCache.append(feature)    # 这样会把feature添加到tagger对象中的featureCache吗？

            feature = []
        
        # 同样，根据二元语法模版编译句子的二元语法特征
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
            
            # 编译特征索引，利用双数组字典树建立特征空间
            id = self.getID(featureID)
            if id != -1:
                feature.append(id)

            #print(f'----cur={curPos}，templ={templ}, featureID={featureID}, id={id}++++')

        return True


    def rebuildFeatures(self, tagger):
        """
        重新编译特征, 在这里重新生成 Node 对象
        -param tagger TaggerImpl 对象
        """
        # 获取当前 taggerImpl 对象（整个句子），句子的初始特征长度（一般等于0）
        # 以及特征缓存（根据语法模版编译的，存储在双数组字典树特征索引）
        fid = tagger.getFeature_id()
        featureCache = tagger.getFeatureCache()
        
        # 句子中每个词的所有特征, 以及每个特征的所有标签，都转换成节点形式
        for cur in range(tagger.size()):
            f = featureCache[fid]
            fid += 1

            for i in range(len(self.y)):
                n = Node()
                n.clear()
                n.x = cur
                n.y = i
                n.fVector = f

                tagger.set_node(n, cur, i)
        
        # 创建路径，但是并未看到路径保存呀？？？？
        # 在p.add() 的时候会把路径添加给节点 -- 2024年7月13号
        for cur in range(1, tagger.size()):
            f = featureCache[fid]
            fid += 1

            for j in range(len(self.y)):
                for i in range(len(self.y)):
                    p = Path()
                    p.clear()
                    p.add(tagger.get_node(cur - 1, j), tagger.get_node(cur, i))
                    p.fvector = f

    def applyRule(self, templ, cur, tagger):
        """
        应用规则，根据特征模版创建特征字符串，会被当作创建特征索引的key
        -param templ 模版特征值, 比如：U3:%x[-2,0]%x[-1,0] 
        -param cur   tagger实例的句子中当前单词的下标
        -param tagger 标记对象，句子
        return 特征字符串
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
            if r != None:
                sb.append(r)

            if len(tuple_) > 1:
                sb.append(tuple_[1])
        
        return ''.join(sb)
            
    @abstractmethod
    def getID(self, featureID):
        pass
    
    @abstractmethod
    def clear(self):
        pass

    def getIndex(self, idxStr, cur, tagger):
        """
        获取标记对象中特定元素的特征索引
        -param idxStr 特征模版中的索引值, 比如：U3:%x[-2,0]%x[-1,0] 中的 [-2,0]
        -param cur tagger实例中语料库的下标
        -param tagger 标记对象, 句子对象
        return 句子中存在的单词
        """
        row, col = int(idxStr[0]), int(idxStr[1])
        pos = row + cur     # 以当前词的位置 cur ，根据 row 向前或者向后获取上下文信息

        if row < -len(FeatureIndex.EOS) or row > len(FeatureIndex.EOS) or col < 0 or col >= tagger.xsize():
            return None

        if self.checkMaxXsize:
            self.max_xsize = max(self.max_xsize, col + 1)
        
        # 设置超前了当前时刻的默认特征，比如：U0:_B-1
        if pos < 0:
            return FeatureIndex.BOS[-pos - 1]
        
        # 设置超出了当前时刻的默认特征，比如：U6:_B+1_B+2 
        elif pos >= tagger.size():
            return FeatureIndex.EOS[pos - tagger.size()]
        
        # 这是直接获取特征的值，比如：无\tB ，直接获取 “无”
        else:
            return tagger.x_str(pos, col)


    def calcCost(self, obj):
        """
        计算损失值（费用）
        """ 
        if isinstance(obj, Path):
            return self._calcCostPath(obj)

        return self._calcCostNode(obj)

    def _calcCostPath(self, path):
        """
        计算转移特征函数的代价
        -param path 边
        """
        path.cost = 0.0
        
        if len(self.alphaFloat) > 0:
            c = 0.0
            i = 0
            
            while path.fvector[i] != -1:
                c += self.alphaFloat[path.fvector[i] + path.lnode.y * len(self.y) + path.rnode.y]
                i += 1

            path.cost = self.costFactor * c

        else:
            c = 0.0
            i = 0
            
            while path.fvector[i] != -1:
                c += self.alpha[path.fvector[i] + path.lnode.y * len(self.y) + path.rnode.y]
                i += 1

            path.cost = self.costFactor * c

        
    def _calcCostNode(self, node):
        """
        计算状态特征函数的代价, 节点的损失值
        -param node 节点
        """
        node.cost = 0.0

        if len(self.alphaFloat) > 0:
            c = 0.0
            i = 0
            
            while node.fVector[i] != -1:
                c += self.alphaFloat[node.fVector[i] + node.y]
                i += 1

            node.cost = self.costFactor * c

        else:
            c = 0.0
            i = 0
            
            # 当前节点所对应的特征向量（节点是特征的所有可能标记）
            # alpha 是在optimizer中计算的，在这之前只是初始化固定的长度，因此感觉这里计算cost也是初始化的过程
            while node.fVector[i] != -1:
                c += self.alpha[node.fVector[i] + node.y]
                i += 1

            node.cost = self.costFactor * c

