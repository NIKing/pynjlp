from nlp.model.perceptron.feature.MutableFeatureMap import MutableFeatureMap
from nlp.model.perceptron.feature.ImmutableFeatureMap import ImmutableFeatureMap

from nlp.model.perceptron.InstanceConsumer import InstanceConsumer

from nlp.model.perceptron.model.StructurePerceptron import StructurePerceptron
from nlp.model.perceptron.model.AveragedPerceptron import AveragedPerceptron

from nlp.corpus.io.IOUtil import loadInstance

class Result():
    def __init__(self, model, prf):
        self.model = model
        self.prf = prf

    def getAccuracy(self):
        if len(self.prf) == 3:
            return self.prf[2]

        return self.prf[0]

    def getModel(self):
        return self.model

class PerceptronTrain(InstanceConsumer):
    
    def train(self, trainingFile, modelFile, developFile = "", compressRatio = 0, maxIteration = 2, threadNum = 0):
        """
        训练
        -param traingFile 训练集文件
        -param developFile 开发集文件
        -param modelFile 模型保存路径
        -param compressRatio 压缩比
        -param maxIteration 最大迭代次数
        -param threadNum 线程数
        return 一个包含模型和精度的结构
        """

        if not developFile:
            developFile = trainingFile
        
        # 创建分词标签集 / 以及标签映射, [B,M,E,S] 
        tagSet = self.createTagSet()
        mutableFeatureMap = MutableFeatureMap(tagSet)
        
        # 分词的特征如何提取? 这个特征是什么样子的？
        print('开始加载训练集...')
        instances = self.loadTrainInstances(trainingFile, mutableFeatureMap)
        print(f'加载完毕，实例一共{len(instances)}句，特征总数{mutableFeatureMap.getSize() * tagSet.size()}')
        
        #print(tagSet.stringIdMap)
        #print(mutableFeatureMap.featureIdMap)

        #print(len(instances[0].sentence), instances[0].sentence)
        #print(len(instances[0].tagArray), instances[0].tagArray)
        #print(len(instances[0].featureMatrix), instances[0].featureMatrix)

        # 开始训练, 如何结构化？如何进行结构预测？
        immutableFeatureMap = ImmutableFeatureMap(mutableFeatureMap.featureIdMap, tagSet = tagSet)
        
        if threadNum == 0:
            model = StructurePerceptron(immutableFeatureMap)
            
            for _iter in range(maxIteration):
                for instance in instances:
                    model.update(instance)

        elif threadNum == 1:
            model = AveragedPerceptron(immutableFeatureMap)
            
            current = 0
            total = [0.0] * len(model.parameter)
            timestamp = [0] * len(model.parameter)

            for _iter in range(maxIteration):
                for instance in instances:
                    current += 1
                    model.update(instance, total, timestamp, current)
            
            # 平均
            model.average(total, timestamp, current)
        
        # 开发集评测
        accuracy = self.evaluate(developFile, model)
        self.printAccuracy(accuracy)

        # 保存模型
        #model.save(modelFile)
        #print('模型保存成功')
        
        return Result(model, accuracy)
    
    def loadTrainInstances(self, trainingFile, mutableFeatureMap):
        """
        加载训练实例
        -param trainingFile 训练集文件
        -param mutableFeatureMap 可变的特征映射对象
        return 实例列表
        """ 

        # 加载文本数据
        corpusList = loadInstance(trainingFile)
        corpusList = corpusList[:int(len(corpusList) * 0.05)]

        # 正规化后/每句话进行特征采集
        instanceList = []
        for sentence in corpusList:
            instanceList.append(self.createInstance(sentence, mutableFeatureMap))

        return instanceList


    def printAccuracy(self, accuracy):
        """打印评测结果"""
        if len(accuracy) == 3:
            print(f"P:{accuracy[0]};R:{accuracy[1]}:;F1:{accuracy[2]}:;")
        else:
            print(f"P:{accuracy[0]}")
