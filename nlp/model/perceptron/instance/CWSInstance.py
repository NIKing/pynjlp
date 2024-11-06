from nlp.model.perceptron.instance.Instance import Instance

"""
感知机分词特征提取器/样本实力化类
"""
class CWSInstance(Instance):
    CHAR_BEGIN = '\u0001'
    CHAR_END = '\u0002'

    def __init__(self, sentence, featureMap):
        super().__init__()

        tagSet = featureMap.tagSet
        
        # 给句子打{B,E,M,S}标签
        tagArray = []
        for sen in sentence:
            
            if len(sen) == 1:
                tagArray.append(tagSet.S)
            else:
                tagArray.append(tagSet.B)
                for k in range(1, len(sen) - 1):
                    tagArray.append(tagSet.M)
                tagArray.append(tagSet.E)
        
        # 注意到了这里，sentence 要变成字符级别的序列
        sentence = ''.join(sentence)

        # 整个句子序列
        self.sentence = sentence

        # 整个句子的标签序列
        self.tagArray = tagArray

        # 整个句子特征矩阵，上下文特征
        self.initFeatureMatrix(sentence, featureMap)

    def initFeatureMatrix(self, sentence, featureMap):
        """
        初始化特征矩阵
        -param sentence 一个训练句子实例
        -param featureMap 特征集合
        """
        self.featureMatrix = []
        for i in range(len(sentence)):
            self.featureMatrix.append(self.extractFeature(sentence, featureMap, i))
   
    def extractFeature(self, sentence, featureMap, position):
        """
        特征提取函数
        -param sentence 一个训练句子实例
        -param featureMap 特征集合
        """
        
        # 下面提取当前单词的上下文标签
        # 句子中，当前单词的上上一个的字符 / 上一个的字符
        pre2Char = sentence[position - 2] if position >= 2 else self.CHAR_BEGIN
        preChar = sentence[position - 1] if position >= 1 else self.CHAR_BEGIN

        curChar = sentence[position]

        # 句子中，当前单词的下下一个的字符 / 下一个的字符
        next2Char = sentence[position + 2] if position < len(sentence) - 2 else self.CHAR_BEGIN
        nextChar = sentence[position + 1] if position < len(sentence) - 1 else self.CHAR_BEGIN

        # 收集当前词的上下文特征列表 
        sbFeatureList = [
            (preChar, '1'), (curChar, '2'), (nextChar, '3'), 
            (pre2Char, '/', preChar, '4'), (preChar, '/', curChar, '5'), 
            (curChar, '/', nextChar, '6'), (nextChar, '/', next2Char, '7')
        ] 
        print(position)
        print(sbFeatureList)

        # 特征向量
        featureVec = []

        # 根据特征列表和特征映射，收集特征向量
        # 注意在这里进行映射转换，featureVec是特征在特征空间中的映射
        for sbFeature in sbFeatureList:
            self.addFeature(sbFeature, featureVec, featureMap)
        
        print(featureVec)
        print('')

        # 返回数组形式
        return self.toFeatureArray(featureVec)
