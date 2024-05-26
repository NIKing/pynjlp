from nlp.model.perceptron.model.LinearModel import LinearModel

"""平均感知机"""
class AveragedPerceptron(LinearModel):
    def __init__(self, featureMap = None, parameter = None):
        super().__init__(featureMap, parameter)

    def update(self, *args):
        """根据答案和预测更新参数"""
        
        if len(args) == 4:
            return self.update_instance(*args)

        if isinstance(args[0], list) and isinstance(args[1], list):
            self.update_function(*args)
        
        elif isinstance(args[0], list):
            self.update_vector(*args)
        
        else:
            self.update_index(*args)

    def update_instance(self, instance, total = list, timestamp = list, current = int):
        if not instance:
            return
        
        # 解码/预测整个句子标签
        guessLabel = [0] * len(instance)
        self.viterbiDecode(instance, guessLabel)
        
        tagSet = self.featureMap.tagSet
        for i in range(len(instance)):
            featureVector = instance.getFeatureAt(i) # 句子中当前单词的上下文向量
            
            goldFeature = [0] * len(featureVector)  # 根据答案应当被激活的特征 
            predFeature = [0] * len(featureVector)  # 实际预测时激活的特征
            
            for j in range(len(featureVector) - 1):
                goldFeature[j] = featureVector[j] * tagSet.size() + instance.tagArray[i]
                predFeature[j] = featureVector[j] * tagSet.size() + guessLabel[i]
            
            # 最后一位，保存的是上一个词标签和当前词标签的关系
            goldFeature[len(featureVector) - 1] = (tagSet.bosId() if i == 0 else instance.tagArray[i - 1]) * tagSet.size() + instance.tagArray[i]
            predFeature[len(featureVector) - 1] = (tagSet.bosId() if i == 0 else guessLabel[i - 1]) * tagSet.size() + guessLabel[i]

            self.update_function(goldFeature, predFeature, total, timestamp, current)


    def update_function(self, goldIndex, predictIndex = list, total = list, timestamp = list, current = int):
        """
        根据答案和预测更新参数
        -param goldIndex            预测正确的特征函数
        -param predictIndex         命中的特征函数
        -param total                权值向量总和
        -param timestamp            每个权值上次更新的时间戳
        -param current              当前时间戳
        """
        for i in range(len(goldIndex)):
            if goldIndex[i] == predictIndex[i]:
                continue

            self.update(goldIndex[i], 1, total, timestamp, current)
            if predictIndex[i] >= 0 and predictIndex[i] < len(self.parameter):
                self.update(predictIndex[i], -1, total, timestamp, current)
            else:
                raise ValueError('更新参数时传入了非法下标')
    
    def update_vector(self, featureVector , value, total, timestamp, current):
        """
        根据特征向量更新参数
        -param featureVector特征向量
        -param value        更新量
        -param total        权值向量总和
        -param timestamp    每个权值上次更新的时间戳
        -param current      当前时间戳
        """
        for i in featureVector:
            self.update_index(i, value, total, timestamp, current)

    def update_index(self, index, value, total, timestamp, current):
        """
        根据特征向量的下标更新参数 
        -param index        特征向量的下标
        -param value        更新量
        -param total        权值向量总和
        -param timestamp    每个权值上次更新的时间戳
        -param current      当前时间戳
        """
        passed = current - timestamp[index]
        
        total[index] += passed * self.parameter[index]
        timestamp[index] = current

        self.parameter[index] += value

    def average(self, total, timestamp, current):
        for i in range(len(self.parameter)):
            self.parameter[i] = (total[i] + (current - timestamp[i]) * self.parameter[i]) / current
        
        return self

    

