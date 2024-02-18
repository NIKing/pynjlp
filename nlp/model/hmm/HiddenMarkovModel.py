import math
import random

import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy

from nlp.utility.Predefine import Predefine 

"""隐马尔可夫模型基类"""
class HiddenMarkovModel(ABC):
    
    def __init__(self, start_probability = None, transition_probability = None, emission_probability = None):
        self.start_probability = deepcopy(start_probability)
        self.transition_probability = deepcopy(transition_probability)
        self.emission_probability = deepcopy(emission_probability)

    def toLog(self):
        """转换张量中概率值为对数概率值"""
        if not self.start_probability or \
                not self.transition_probability or \
                not self.emission_probability:
                    return None
        
        for i in range(len(self.start_probability)):
            #self.start_probability[i] = math.log(max(Predefine.MIN_PROBABILITY, self.start_probability[i]))

            if self.start_probability[i] == 0.0:
                self.start_probability[i] = float('-inf')
            else:
                self.start_probability[i] = math.log(self.start_probability[i])


            for j in range(len(self.transition_probability[i])):
                #self.transition_probability[i][j] = math.log(max(Predefine.MIN_PROBABILITY, self.transition_probability[i][j]))

                if self.transition_probability[i][j] == 0.0:
                    self.transition_probability[i][j] = float('-inf')
                else:
                    self.transition_probability[i][j] = math.log(self.transition_probability[i][j])

            for j in range(len(self.emission_probability[i])):
                #self.emission_probability[i][j] = math.log(max(Predefine.MIN_PROBABILITY, self.emission_probability[i][j]))
                
                if self.emission_probability[i][j] == 0.0:
                    self.emission_probability[i][j] = float('-inf')
                else:
                     self.emission_probability[i][j] = math.log(self.emission_probability[i][j])

    
    def unLog(self):
        """将对数概率值转回原值"""
        for i in range(len(self.start_probability)):
            self.start_probability[i] = math.exp(self.start_probability[i])

        for i in range(len(self.emission_probability)):
            for j in range(len(self.emission_probability[i])):
                self.emission_probability[i][j] = math.exp(self.emission_probability[i][j])

        for i in range(len(self.transition_probability)):
            for j in range(len(self.transition_probability[i])):
                self.transition_probability[i][j] = math.exp(self.transition_probability[i][j])

    def logToCdf(self, log):
        """
        对数概率值转为累计分布函数
        之所以需要用到累计分布，是因为下面需要使用二分法进行采样
        累计分布数据最大值等于1，在采样过程中使用随机数也是在（0，1）区间内
        """
        if not log:
            return []

        if np.ndim(log) == 1: 
            cdf = [0.0] * len(log)
            
            # 注意，计算的是累计值，其中Math.exp()是获取以e为自然底数，
            cdf[0] = math.exp(log[0])
            for i in range(1, len(cdf) - 1):
                cdf[i] = cdf[i - 1] + math.exp(log[i])

            cdf[len(cdf) - 1] = 1.0

        else:

            cdf = [[0.0] for i in range(len(log))]
            for i in range(len(cdf)):
                cdf[i] = self.logToCdf(log[i])
        
        return cdf
    
    def drawFrom(self, cdf) -> int:
        """
        采样, 根据（0,1）之间的随机数，在累计概率分布区间进行采样
        -param cdf 概率分布区间
        return int
        """
        index = np.searchsorted(cdf, random.random(), side = 'left')
        
        if index >= 0:
            return index
        else:
            return -index - 1
    
    def generateSamples(self, minLength, maxLength, size):
        """
        生成样本序列
        -param minLength 样本最低长度
        -param maxLength 样本最高长度
        -param size 需要生成的样本数量
        return 样本序列集合
        """
        samples = [] * size
        for i in range(size):
            samples.append(self.generate((round(random.random() * (maxLength - minLength)) + minLength)))
        return samples
    
    @abstractmethod
    def generate(self, length):
        pass
    
    @abstractmethod
    def predict(self, observation, state) -> float:
        """
        预测 - 解码 - 通过维特比算法在有向无环图上计算最长路径
        -param observation 观测序列
        -param state 预测状态序列, 默认0，需要赋值返回
        return 概率对数
        """
        pass

    def train(self, samples):
        """
        训练 - 编码 - 通过统计方法计算数据集中初始概率向量、转移概率矩阵和发射概率矩阵
        -param samples 数据集 int[i][j] ; i = 0 为观测序列；i = 1为状态序列，j为时序轴
        """
        if not samples:
            return None
        
        # 获取样本中最大显/隐状态值
        max_state, max_obser = 0, 0
        for sample in samples:
            if len(sample) != 2 or len(sample[0]) != len(sample[1]):
                continue

            for o in sample[0]:
                max_obser = max(max_obser, o)

            for s in sample[1]:
                max_state = max(max_state, s)

        self.estimateStartProbability(samples, max_state)
        self.estimateTransitionProbability(samples, max_state)
        self.estimateEmissionProbability(samples, max_state, max_obser)

        self.toLog()

    def estimateStartProbability(self, samples, max_state):
        """
        估计（统计）初始概率向量
        -param samples 样本数据
        -param max_state 状态序列的最大下标
        """

        # 统计 samples 中第二行中第一个元素出现的次数，也即隐藏状态的初始频次
        start_probability = [0.0] * (max_state + 1)
        for sample in samples:
            s = sample[1][0]
            start_probability[s] += 1 
        
        self.normalize(start_probability)
        self.start_probability = start_probability

    def estimateTransitionProbability(self, samples, max_state):
        """
        利用极大似然估计（统计）转移概率矩阵
        -param samples 样本数据
        -param max_state 状态序列的最大下标
        """
        transition_probability = np.zeros((max_state + 1, max_state + 1)).tolist()
        for sample in samples:
            prev_s = sample[1][0]

            for i in range(1, len(sample[1])):
                s = sample[1][i]
                transition_probability[prev_s][s] += 1

                prev_s = s
        
        # 归一化处理
        for transition in transition_probability:
            self.normalize(transition)
        
        self.transition_probability = transition_probability 

    def estimateEmissionProbability(self, samples, max_state, max_obser):
        """
        估计（统计）发射概率矩阵
        -param samples 样本数据
        -param max_state 状态序列的最大下标
        -param max_obser 观察序列的最大下标
        """
        emission_probability = np.zeros((max_state + 1, max_obser + 1)).tolist()
        for sample in samples:
            for i in range(len(sample[0])):
                o = sample[0][i]
                s = sample[1][i]

                emission_probability[s][o] += 1
        
        for emission in emission_probability:
            self.normalize(emission)

        self.emission_probability = emission_probability

    def normalize(self, freq):
        """
        频次归一化处理
        """
        _sum = sum(freq)
        for i in range(len(freq)):
            freq[i] /= _sum
    
    def similar(self, model) -> bool:
        """对比两个模型的相似度"""
        if not HiddenMarkovModel.static_similar(self.start_probability, model.start_probability):
            return False
        
        for i in range(len(self.transition_probability)):
            if not HiddenMarkovModel.static_similar(self.transition_probability[i], model.transition_probability[i]):
                return False
            
            if not HiddenMarkovModel.static_similar(self.emission_probability[i], model.emission_probability[i]):
                return False

        return True
    
    @staticmethod
    def static_similar(A, B):
        eta = 1e-2 # 0.01
        for i in range(len(A)):
            if abs(A[i] - B[i]) > eta:
                return False

        return True

