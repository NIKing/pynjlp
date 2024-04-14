from nlp.model.perceptron.common.TaskType import TaskType
from nlp.model.perceptron.tagset.TagSet import TagSet

from nlp.model.perceptron.feature.LockableFeatureMap import LockableFeatureMap

from nlp.model.perceptron.model.LinearModel import LinearModel
from nlp.model.perceptron.model.AveragedPerceptron import AveragedPerceptron

from abc import ABC, abstractmethod

import pandas as pd

"""样本"""
class Instance():
    def __init__(self, x, y):
        """
        -param x 特征向量
        -param y 标签
        """
        self.x = x
        self.y = y


"""感知机基类-二分类"""
class PerceptronClassifier(ABC):
    def __init__(self, model = None):

        if model != None and model.taskType() != TaskType.CLASSIFICATION:
            return "模型不是分类模型"
        
        if isinstance(model, str):
            self.model = LinearModel(modelFile = model)
        else:
            self.model = model


    def train(self, corpus, maxIteration = 2, averagePerceptron = True):
        """
        感知机训练器
        -param corpus 语料库
        -param maxIteration 最大迭代次数
        -param averagePerceptron 是否使用平均感知器
        """

        # 特征映射
        featureMap = LockableFeatureMap(TagSet(TaskType.CLASSIFICATION))
        featureMap.mutable = True
        
        # 读取实例，根据不同选项训练模型
        instanceList = self.readInstance(corpus, featureMap)
         
        if averagePerceptron:
            self.model = self.trainAveragedPerceptron(instanceList, featureMap, maxIteration)
        else:
            self.model = self.trainNaivePerceptron(instanceList, featureMap, maxIteration)
        
        # 训练后，特征不可写
        featureMap.mutable = False

        return self.evaluate(instanceList)


    def trainNaivePerceptron(self, instanceList, featureMap, maxIteration):
        """
        朴素感知机训练算法
        -param instancelist 训练实例列表
        -param feqturemap 特征函数
        -param maxinteration 最大训练次数
        """
        model = LinearModel(featureMap, [0.0] * len(featureMap))
        for it in range(maxIteration):
            for instance in instanceList:
                y = model.decode(instance.x)

                if y != instance.y:
                    model.update(instance.x, instance.y)

        return model


    def trainAveragedPerceptron(self, instanceList, featureMap, maxIteration):
        """
        平均感知机训练算法
        -param instancelist 训练实例列表
        -param feqturemap 特征函数
        -param maxinteration 最大训练次数
        """
        parameter, _sum, time = [0.0] * len(featureMap), [0.0] * len(featureMap), [0.0] * len(featureMap)
        model = AveragedPerceptron(featureMap, parameter)
        
        t = 0
        for it in range(maxIteration):

            for instance in instanceList:
                t += 1
                
                # 预测，对比预测结果和标准结果，若不正确则更新模型
                y = model.decode(instance.x)
                if y != instance.y:
                    model.update(instance.x, instance.y, _sum, time, t)


        return model.average(_sum, time, t)

    def readInstance(self, corpus, featureMap):
        """
        从语料库读取实例, 在这里调用不同特征函数获取特征值
        -param corpus 语料库
        -param featureMap 特征映射
        return 数据集
        """
        instanceList = []
        lineIterator = pd.read_csv(corpus).loc[:,[True, True]].values
        
        #lineIterator = lineIterator[:2]
        print(f'训练条目总数:{len(lineIterator)}')
        for line in lineIterator:
            text, label = line
            
            x = self.extractFeature(text, featureMap)
            y = featureMap.tagSet.add(label)

            if y == 0:
                y = -1
            elif y > 1:
                raise ValueError("类别数目大于2，目前只支持二分类")

            instanceList.append(Instance(x, y))

        return instanceList

    @abstractmethod
    def extractFeature(self, text, featureMap) -> list:
        """
        特征提取
        -param text 文本
        -param featureMap 特征映射
        return 特征向量
        """
        pass

    def addFeature(self, feature, featureMap, featureList):
        """
        向特征向量插入特征
        -param str  feature 特征
        -param map  featureMap 特征映射, 由双数组树构成
        -param list featureList 特征向量
        """
        featureId = featureMap.idOf(feature)
        if featureId != -1:
            featureList.append(featureId)

    def predict(self, text):
        """预测"""
        y = self.model.decode(self.extractFeature(text, self.model.featureMap))
        if y == -1:
            y = 0

        return self.model.tagSet().stringOf(y)

    def evaluate(self, instanceList):
        """
        模型评估
        -param instanceList 实例列表，当类型是路径，就需要加载成实例对象
        """
        if isinstance(instanceList, str):
            instanceList = self.readInstance(instanceList, self.model.featureMap)

        TP, FP, FN = 0, 0, 0,
        for instance in instanceList:
            y = self.model.decode(instance.x)
            if y == 1:
                if instance.y == 1:
                    TP += 1
                else:
                    FP += 1
            elif instance.y == 1:
                FN += 1

        p = TP / (TP + FP) * 100
        r = TP / (TP + FN) * 100

        return p, r, 2 * p * r / (p + r)
    
    def getModel(self):
        return self.model
