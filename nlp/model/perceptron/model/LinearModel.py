from nlp.model.perceptron.feature.ImmutableFeatureMDatMap import ImmutableFeatureMDatMap
from nlp.model.perceptron.feature.FeatureSortItem import FeatureSortItem

from nlp.model.perceptron.common.TaskType import TaskType

from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger

from nlp.algorithm.MaxHeap import MaxHeap
from nlp.utility.MathUtility import MathUtility
from nlp.utility.Predefine import Predefine

from nlp.corpus.io.IOUtil import writeListToBin, writeTxtByList
from nlp.corpus.io.ByteArrayStream import ByteArrayStream

"""线性模型-基础模型"""
class LinearModel():

    def __init__(self, featureMap = None, parameter = None, modelFile = ""):
        """
        初始化
        -param featureMap 特征函数
        -param parameter 特征权重
        -modelFile 模型路径
        """
        if parameter == None and featureMap:
            # 特征编号最大值 * 标签编号最大值，可以保证后续计算时所使用占用的空间
            # 在结构化感知机中，通过特征映射编号 * 标签集数量 + 当前词的标签编号来表示特征索引，因此，在这里使用 * tagSet.size() 来保证上一个词在【BMES】结果上都有位置.
            parameter = [0.0] * (featureMap.getSize() * featureMap.tagSet.size())
        
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
            self.featureMap = ImmutableFeatureMDatMap(featureIdMap = self.featureMap.entrySet(), tagSet = self.tagSet())
        
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
            pair = []
            bw.append(entry.getKey())
            
            #print(entry.getKey(), entry.getValue())

            values = []
            # 双数组中保存的键值对数量和权重数量一致，一般情况是一致的
            # 在提取特征的时候，会把提取到的特征当作key，把当前动态数组大小当作value，组合成键值对存放在动态双数组字典树上。
            # 当提取的特征数据是多个的时候，特征函数返回的是特征向量，这些特征向量都具有相同的标签
            # 进行在线学习的时候（执行update），会用特征向量中的特征值在字典树的value作为parameter的key保存，并依次进行更新权重数据。
            # 因此，在这里可以直接从parameter取特征的权重值
            if entry.size() == len(self.parameter):
                values = str(self.parameter[entry.getValue()])
            else:
                for i in range(len(tagSet)):
                    values.append(str(self.parameter[entry.getValue() * tagSet.size() + i]))
            
            bw.append(''.join(''.join(values)))
        
        #print(tagSet.stringIdMap)
        #print(bw)
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
    
    def load(self, modelFile):
        if not modelFile:
            return 
        
        # 读取模型
        byteArray = ByteArrayStream(modelFile)
        
        # 初始化特征映射对象
        self.featureMap = ImmutableFeatureMDatMap()
        self.featureMap.load(byteArray)
        
        size = self.featureMap.getSize()
        tagSet = self.featureMap.tagSet
        
        # 加载特征值
        if tagSet.taskType == TaskType.CLASSIFICATION:
            parameter = byteArray.next(count = size)

        else:
            parameter = [0.0] * (size * tagSet.size())
            for i in range(size):
                for j in range(tagSet.size()):
                    parameter[i * tagSet.size() + j] = byteArray.next()
        
        self.parameter = parameter


    def viterbiDecode(self, instance, guessLabel):
        """
        维特比算法解码
        -param instance 样本实例
        -param guessLabel 预测标签
        """
        
        # 所有的标签（编号）/ bos初始标记（标签的数量）
        allLabel = self.featureMap.allLabels()
        bos = self.featureMap.bosTag()

        sentenceLength = len(instance.tagArray)
        labelSize = len(allLabel)
        
        # 句子中每个字符的标签预测/
        preMatrix = [[0] * labelSize] * sentenceLength
        scoreMatrix = [[0.0] * labelSize] * 2
        
        #print('===', allLabel)
        #print('---', scoreMatrix)
        # 前向遍历
        for i in range(sentenceLength):
            
            # 按位与，i & 1 表示与最低位(最右位)计算为1，通过此方法判断当前 i 的奇偶性
            _i = i & 1      
            _i_1 = 1 - _i

            #print(f'i={i}={_i}')

            allFeature = instance.getFeatureAt(i)
            transitionFeatureIndex = len(allFeature) - 1 # 承上启下，代表句子中上一个特征向量的标签
            
            # 首次转移特征
            if i == 0:
                allFeature[transitionFeatureIndex] = bos
                for j in range(len(allLabel)):
                    preMatrix[0][j] = j
                    
                    score = self.score(allFeature, j)
                    scoreMatrix[0][j] = score
                
                continue
            
            # N * N 转移特征, 获取句子中前后两个词隐状态最大转移得分标签，前一个词的【BMES】标签到后一个词【BMES】标签，最大分数和标签
            # preMatrix 是 M * N，也就是句子中每个词，在【B,M,E,S】标签集的最大概率标签（注意，N的位置存放的是最大概率的标签）
            # scoreMatrix 是 2 * N，它是一个临时变量，通过计算句子位置的奇偶性，来保存句子前后两个词，在【B,M,E,S】标签集的最大分数
            for curLabel in range(len(allLabel)):
                maxScore = Predefine.INTEGER_MIN_VALUE

                for preLabel in range(len(allLabel)):
                    
                    # 转移特征 得到 特征向量
                    allFeature[transitionFeatureIndex] = preLabel
                                       
                    # 也是联合概率？？？上一个分数和当前分数
                    score = self.score(allFeature, curLabel)
                    curScore = scoreMatrix[_i_1][preLabel] + score
                    
                    if maxScore < curScore:
                        maxScore = curScore
                        
                        preMatrix[i][curLabel] = preLabel
                        scoreMatrix[_i][curLabel] = maxScore

        
        # 获取最后一个词的分数和标签索引
        maxIndex = 0
        maxScore = scoreMatrix[(sentenceLength - 1) & 1][0]
        for index in range(len(allLabel)):
            if maxScore < scoreMatrix[(sentenceLength - 1) & 1][index]:
                maxIndex = index
                maxScore = scoreMatrix[(sentenceLength - 1) & 1][index]
        
        # 向后回溯
        for i in range(sentenceLength - 1, 0, -1):
            guessLabel[i] = allLabel[maxIndex]
            maxIndex = preMatrix[i][maxIndex]

        return maxScore
    
    def score(self, featureVector, currentTag):
        """
        打分函数，特征向量与权重向量做点积之后得到的一个标量，w * \phi(x, y)
        -param featureVector 特征id构成的特征向量，与下面currentTag一致，同属于句子某个字符的特征向量
        -param currentTag 当前标签, 与上面featureVector一致，同属于句子某个字符的标签ID
        return 当前字符属于某个标签（状态）下的特征向量的累积分数 
        """
        score = 0
        for index in featureVector:
            if index == -1:
                continue

            if index < -1 or index >= self.featureMap.getSize():
                raise ValueError('在打分时传入了非法的下标')

            index = index * self.featureMap.tagSet.size() + currentTag
            score += self.parameter[index]  # 累计特征权重

        return score
    

