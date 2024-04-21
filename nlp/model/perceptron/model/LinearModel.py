from nlp.model.perceptron.feature.ImmutableFeatureMDatMap import ImmutableFeatureMDatMap
from nlp.model.perceptron.feature.FeatureSortItem import FeatureSortItem

from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger

from nlp.algorithm.MaxHeap import MaxHeap
from nlp.utility.MathUtility import MathUtility
from nlp.corpus.io.IOUtil import writeListToBin, writeTxtByList

"""线性模型-基础模型"""
class LinearModel():

    def __init__(self, featureMap = None, parameter = None, modelFile = ""):
        """
        初始化
        -param featureMap 特征函数
        -param parameter 特征权重
        -modelFile 模型路径
        """

        self.featureMap = featureMap
        self.parameter = parameter
        
        if modelFile != "":
            self.load(modelFile)

    def decode(self, x):
        """
        分离超平面解码
        -param x 特征向量
        return sign(x)
        """
        y = 0
        for f in x:
            y += self.parameter[f]

        return -1 if y < 0 else 1

    def update(self, x, y):
        """
        参数更新
        -param x 特征向量
        -param y 正确答案
        """
        assert y == 1 or y == -1 , "感知机标签y必须是±1"

        for f in x:
            self.parameter[f] += y
    
    def tagSet(self):
        return self.featureMap.tagSet

    def taskType(self):
        return self.featureMap.tagSet.taskType

    def save(self, modelFile, featureIdSet = None, ratio = 0, text = False):
        """
        保存模型
        -param modelFile    路径
        -param featureIdSet 特征集
        -param ratio        压缩比
        -param text         是否输出文本以供调试
        """
        #self.compress(ratio, 1e-3)
        
        # 保存到二进制
        self.saveToBin(modelFile)
        
        # 保存到文本
        if not text:
            return

        if not featureIdSet:
            featureIdSet = self.featureMap.entrySet()
    
        self.saveToText(modelFile, featureIdSet)


    def saveToBin(self, modelFile):
        """保存模型二进制数据到.bin"""
        if not isinstance(self.featureMap, ImmutableFeatureMDatMap):
            self.featureMap = ImmutableFeatureMDatMap(self.featureMap.entrySet(), self.tagSet())
        
        # 保存特征映射值和特征权重
        out = []
        
        # 保存双数组
        self.featureMap.save(out)

        # 保存权重参数, 数组的位置代表特征映射，位置中的值表示权重
        for aParameter in self.parameter:
            out.append(aParameter)
        
        writeListToBin(modelFile, out) 

    def saveToText(self, modelFile, featureIdSet = None):
        """为了调试方便，保存模型数据到.txt"""
        if not featureIdSet:
            return
        
        bw = []
        tagSet = self.tagSet()
        
        for entry in featureIdSet:
            #print(entry.getKey(), entry.getValue())
            
            pair = []
            bw.append(entry.getKey())
            
            values = []
            if entry.size() == len(self.parameter):
                values = str(self.parameter[entry.getValue()])
            else:
                for i in range(len(tagSet)):
                    values.append(str(self.parameter[entry.getValue() * tagSet.size() + i]))
            
            bw.append(''.join(''.join(values)))

        writeTxtByList(modelFile + '.txt', bw)
    
    def compress(self, ratio = 0, threshold = 1e-3):
        """
        压缩权重参数空间和双数组空间
        -param ratio 压缩比c（压缩掉的体积，压缩后体积变成1-c）
        -param threshold 特征权重绝对值之和最低阈值
        """
        if ratio < 0 or ratio >= 1:
            raise ValueError('压缩比必须介于 0 和 1 之间')

        if ratio == 0:
            return self

        featureIdSet = self.featureMap.entrySet()
        tagSet = self.tagSet()

        heap = MaxHeap((featureIdSet.size() - tagSet.sizeIncludingBos()) * (1 - ratio), lambda o1, o2 : o1.total - o2.total)

        print('裁剪特征...')
        logEvery = math.ceil(self.featureMap.size() / 10000)
        n = 0
        for entry in featureIdSet:
            n += 1
            if n % logEvery == 0 or self.featureMap.size():
                print('\r%.2%%', MathUtility.percentage(n, self.featureMap.size()))

            if entry.getValue() < tagSet.sizeIncludingBos():
                continue

            item = FeatureSortItem(entry, self.parameter, tagSet.size())
            if item.total < threshold:
                continue

        print('\n裁剪完毕\n')
        
        # 构建双数组字典树
        size = heap.size() + tagSet.sizeIncludingBos()
        parameter = [1.0] * (size + tagSet.size())

        mdat = MutableDoubleArrayTrieInteger()
        for tag in tagSet:
            mdat.add("BL=" +  tag.getKey())

        mdat.add("BL=_BL_")
        for i in range(tagSet.size() * tagSet.sizeIncludingBos()):
            parameter[i] = self.parameter[i]

        n = 0
        for item in heap:
            n += 1
            if n % logEvery == 0 or n == heap.size():
                print('\r%.2%%', MathUtility.percentage(n, heap.size()))

            id = mdat.size()
            mdat.put(item.key, id)

            for i in range(tagSet.size()):
                parameter[id * tagSet.size() + i] = self.parameter[item.id * tagSet.size() + i]


        self.featureMap = ImmutableFeatureMDatMap(mdat, tagSet)
        self.parameter  = parameter

        return self












