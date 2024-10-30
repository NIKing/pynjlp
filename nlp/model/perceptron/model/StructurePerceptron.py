from nlp.model.perceptron.model.LinearModel import LinearModel

"""
结构化感知模型
"""
class StructurePerceptron(LinearModel):
    def __init__(self, featureMap, parameter = None):
        super().__init__(featureMap, parameter)

    def update(self, instance):
        """
        在线学习
        -param instance 一个句子的样本实例
        """ 
        # 解码/预测整个句子标签
        guessLabel = [0] * len(instance)
        self.viterbiDecode(instance, guessLabel)
        
        tagSet = self.featureMap.tagSet
        for i in range(len(instance)):
            featureVector = instance.getFeatureAt(i) # 句子中当前单词的上下文向量
            
            goldFeature = [0] * len(featureVector)  # 根据答案应当被激活的特征 
            predFeature = [0] * len(featureVector)  # 实际预测时激活的特征
            
            # 在这里长度 - 1，因为在Instance.py中，特意多添加了一列，且在最后补充了最后一个特征索引值
            
            # 下面是模拟隐马尔可夫模型的转移概率，k = i * N + j, 其中 k = 特征索引；N 表示标注集大小；i 表示特征向量第i个特征的标签；j 表示样本实例第j 个单词的标签

            # 这里有个问题，若 i 表示样本句子中第i个词语的，那么tagArray是表示句子每个字符的数组，下面的计算就不合理
            # 问题得以解决，tagArray是字符级别的序列，featureVector同样也是字符级别的特征向量，之前的代码有问题
            
            # 找到当前中心词的预测特征向量和标准特征向量，预测特征向量是根据维特比解码而来，标准特征向量是在实例对象上
            # 不仅如此，标准特征向量是以当前字符的正确的标记，从当前字符的特征矩阵从提取出来的，意思就是下面的goldFeature 是当前字符特征矩阵中，全是B 或 全是A 的向量-- 2024年10月30日
            for j in range(len(featureVector) - 1):
                goldFeature[j] = featureVector[j] * tagSet.size() + instance.tagArray[i]
                predFeature[j] = featureVector[j] * tagSet.size() + guessLabel[i]
            
            # 最后一位，保存的是上一个词标签和当前词标签的关系
            goldFeature[len(featureVector) - 1] = (tagSet.bosId() if i == 0 else instance.tagArray[i - 1]) * tagSet.size() + instance.tagArray[i]
            predFeature[len(featureVector) - 1] = (tagSet.bosId() if i == 0 else guessLabel[i - 1]) * tagSet.size() + guessLabel[i]

            self._update(goldFeature, predFeature)


    def _update(self, goldIndex, predictIndex):
        """
        在线学习
        -param goldIndex 标准答案特征函数
        -param predictIndex 预测答案特征函数
        """
        for i in range(len(goldIndex)):

            if goldIndex[i] == predictIndex[i]:
                continue

            # 预测结果与标准结果不对，则奖励正确的特征函数（标准特征函数，将它的权重值加1）
            self.parameter[goldIndex[i]] += 1
            
            # 同时惩罚预测错误的特征函数
            if predictIndex[i] >= 0 and predictIndex[i] < len(self.parameter):
                self.parameter[predictIndex[i]] -= 1


