from abc import ABC, abstractmethod

from nlp.corpus.io.IOUtil import loadInstance
from nlp.dictionary.other.CharTable import CharTable

class InstanceConsumer(ABC):
    tableChar = []
    
    def __init__(self):
        pass
        #print(len(CharTable.CONVERT))

        #InstanceConsumer.tableChar = CharTable.CONVERT
        #for c in range(32):
        #    InstanceConsumer.tableChar[c] = '&'

    @abstractmethod
    def createInstance(self, sentence, mutableFeatureMap):
        pass

    def evaluate(self, developFile, model) -> float:
        """
        模型预测
        -param developFile 测试集样本
        -param model 模型
        return int
        """
        if not developFile or not model:
            return []
        
        corpusList = loadInstance(developFile)
        corpusList = corpusList[int(len(corpusList) * 0.99):]
    
        print('评测的数量=', len(corpusList))

        stat = [0] * 2
        for sentence in corpusList:
            instance = self.createInstance(sentence, model.featureMap)
            self.accuracy(instance, model, stat)

        return [(stat[1] / stat[0]) * 100]


    def accuracy(self, instance, model, stat):
        """
        计算句子的精确度, 这里统计的是句子中每个词预测结果中预测正确的标签数量
        -param instance 单个句子的实例对象
        -param model 模型
        -param stat 统计值列表 [句子长度，预测准确数量]
        """
        predLabel = [0] * len(instance)
        model.viterbiDecode(instance, predLabel)
        
        # 统计句子标签的精确数量和总数量
        stat[0] += len(instance.tagArray)
        for i, pre in enumerate(predLabel):
            
            if pre == instance.tagArray[i]:
                stat[1] += 1

