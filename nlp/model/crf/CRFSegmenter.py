from nlp.NLP import NLPConfig

from nlp.model.crf.CRFTagger import CRFTagger
from nlp.model.crf.crfpp.FeatureIndex import FeatureIndex

from nlp.model.perceptron.instance.CWSInstance import CWSInstance
from nlp.model.perceptron.PerceptronSegmenter import PerceptronSegmenter

from nlp.dictionary.other.CharTable import CharTable
from nlp.corpus.io.IOUtil import loadInstance

"""训练实例"""
class CRFInstance(CWSInstance):
    def __init__(self, text, featureMap, featureTemplateArray):
        self.featureTemplateArray = featureTemplateArray
        super().__init__(text, featureMap)

    def extractFeature(self, sentence, featureMap, position):
        """
        提取特征函数
        -param sentence 句子
        -param featureMap 特征空间
        -param position 句子索引
        """
        sbFeature, featureVec = [], []
        for i in range(len(self.featureTemplateArray)):
            offsetIterator = self.featureTemplateArray[i].offsetList
            delimiterIterator = self.featureTemplateArray[i].delimiterList

            for j, offset in enumerate(offsetIterator):
                offset = offset[0] + position

                if offset < 0:
                    sbFeature.append(FeatureIndex.BOS[-(offset) + 1])
                elif offset >= len(sentence):
                    sbFeature.append(FeatureIndex.EOS[offset - len(sentence)])
                else:
                    sbFeature.append(sentence[offset])

                if delimiterIterator[j + 1]:
                    sbFeature.append(delimiterIterator[j + 1])
                else:
                    sbFeature.append(i)

            self.addFeatureTheClear(sbFeature, featureVec, featureMap)

        return self.toFeatureArray(featureVec)


"""分词类"""
class CRFSegmenter(CRFTagger):
    def __init__(self, modelPath = ""):

        # 不能使用 not modelPath，只有当为默认值的时候，代表可以使用默认模型路径
        if modelPath == "":
            modelPath = NLPConfig.CRFCWSModelPath 

        super().__init__(modelPath)
        
        # 在crf中，用到了感知机分词器, 并把CRF的线性模型带入进去了
        self.proceptronSegmenter = None
        if modelPath:
            self.proceptronSegmenter = PerceptronSegmenter(self.model)
    
    def segment(self, text, normalized = "", wordList = None):
        if not normalized:
            normalized = CharTable.convert(text)
        
        if wordList == None:
            wordList = []

        self.proceptronSegmenter.segment(text, self.createInstance(normalized), wordList)

        return wordList


    def createInstance(self, text):
        """返回的是结构化感知机的分词实例，CRFInstance继承CWSInstance"""
        featureTemplateArray = self.model.getFeatureTemplateArray()
        return CRFInstance(text, self.model.freatureMap, featureTemplateArray)
    
    def convertCorpus(self, filePath) -> list:
        """转换【BMES】标记"""
        sentences = loadInstance(filePath)
        
        bw = []
        for sentence in sentences:
            
            for word in sentence.toSimpleWordList():
                word = CharTable.convert(word.value)

                if len(word) == 1:
                    bw.append(''.join([word, "\t", "S"]))
                    continue

                bw.append(''.join([word[0], "\t", "B"]))
                
                for c in word[1: len(word) - 1]:
                    bw.append(''.join([c, "\t", "M"]))
                
                bw.append(''.join([word[-1], "\t", "E"]))
        
            bw.append("\t")

        return bw

    def getDefaultTemplateData(self):
        """
        一元语法和二元语法模型模版索引
        # 表示开头
        B 表示转移特征
        UO,U1表示特征编号
        x[-1,0]表示一元语法特征
        x[-2,0]%[-1,0]表示二元语法特征
        % 表示为分隔符，可以用任意符号表示，或者去掉分隔符
        这个就是结构化感知机中特征提取，根据当前词，提取上下文两个词的特征。对每个时刻都进行一次提取，当前时刻为t = 0, 过去时刻为负数，未来时刻为正数。例如，[-1, 0]表示上一个字符一元语法特征，[0,0]表示当前字符一元语法特征
        """
        return ['# Unigram', 'U0:%x[-1,0]', 'U1:%x[0,0]', 'U2:%x[1,0]', 'U3:%x[-2,0]%x[-1,0]', 'U4:%x[-1,0]%x[0,0]', 'U5:%x[0,0]%x[1,0]', 'U6:%x[1,0]%x[2,0]', '# Bigram', 'B']


