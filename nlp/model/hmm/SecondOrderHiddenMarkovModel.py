import math
import numpy as np

from nlp.model.hmm.HiddenMarkovModel import HiddenMarkovModel
from nlp.utility.Predefine import Predefine 

"""二阶隐马尔可夫模型"""
class SecondOrderHiddenMarkovModel(HiddenMarkovModel):
    
    def __init__(self, start_probability = None, transition_probability = None, emission_probability = None, transition_probability2 = None):
        """
        -param start_probability 初始状态概率向量
        -param transition_probability 转移状态概率矩阵
        -param emission_probability 发射状态概率矩阵
        """
        super().__init__(start_probability, transition_probability, emission_probability)
        self.transition_probability2 = transition_probability2

        self.toLog()

    def generate(self, length):
        """
        具体采样算法
        -param length 采样长度
        return [] * length
        """

        # 计算三元组的累计概率分布
        pi = self.logToCdf(self.start_probability)
        A  = self.logToCdf(self.transition_probability)
        A2 = self.logToCdf(self.transition_probability2)
        B  = self.logToCdf(self.emission_probability)

        xy = [[0 for j in range(length)] for i in range(2)]
        
        # 二阶模型中，第一、二个状态还是转移矩阵，因此单独计算
        xy[1][0] = self.drawFrom(pi)             # 根据初始状态概率，采样首个隐状态
        xy[0][0] = self.drawFrom(B[xy[1][0]])    # 根据发射概率矩阵，采样首个隐状态对应的首个显状态
        
        xy[1][1] = self.drawFrom(A[xy[1][0]])
        xy[0][1] = self.drawFrom(B[xy[1][1]])
        
        # 第三个状态变成了三维张量
        for t in range(2, length):
            xy[1][t] = self.drawFrom(A2[xy[1][t-2]][xy[1][t-1]])   # 根据转移概率矩阵，继续采样，补充到状态序列
            xy[0][t] = self.drawFrom(B[xy[1][t]])   # 根据发射概率矩阵，继续采样，补充到观察序列

        return xy
    
    def estimateTransitionProbability(self, samples, max_state):
        """
        利用极大似然估计（统计）转移概率矩阵
        -param samples 样本数据
        -param max_state 状态序列的最大下标
        """
        transition_probability  = np.zeros((max_state + 1, max_state + 1)).tolist()
        transition_probability2 = np.zeros((max_state + 1, max_state + 1, max_state + 1)).tolist()

        for sample in samples:
            prev_s = sample[1][0]
            prev_prev_s = -1

            for i in range(1, len(sample[0])):
                s = sample[1][i]
                if i == 1:
                    transition_probability[prev_s][s] += 1
                else:
                    transition_probability2[prev_prev_s][prev_s][s] += 1

                prev_prev_s = prev_s
                prev_s = s
        
        for p in transition_probability:
            self.normalize(p)

        for pp in transition_probability2:
            for p in pp:
                self.normalize(p)
        
        self.transition_probability  = transition_probability
        self.transition_probability2 = transition_probability2


    def toLog(self):
        super().toLog()

        if not self.transition_probability2:
            return None

        for m in self.transition_probability2:
            for v in m:
                for i in range(len(v)):
                    if v[i] == 0.0:
                        v[i] = float('-inf')
                    else:
                        v[i] = math.log(v[i])

    def unLog(self):
        super().unLog()

        for m in self.transition_probability2:
            for v in m:
                for i in range(len(v)):
                    v[i] = math.exp(v[i])

    
    def predict(self, observation, state):
        time, max_s = len(observation), len(self.start_probability)
        score, first = np.zeros((max_s, max_s)), np.zeros(max_s)

        link = np.zeros((time, max_s, max_s), dtype=int)
        for cur_s in range(max_s):
            first[cur_s] = self.start_probability[cur_s] + self.emission_probability[cur_s][observation[0]]

        if time == 1:
            best_s, max_score = 0, Predefine.INTEGER_MIN_VALUE
            for cur_s in range(max_s):
                if first[cur_s] > max_score:
                    best_s = cur_s
                    max_score = first[cur_s]

            state[0] = best_s
            return max_score

        # 第二个时刻，使用前一个时刻的概率向量 * 一阶转移矩阵 * 发射概率矩阵 
        for f in range(max_s):
            for s in range(max_s):
                p = first[f] + self.transition_probability[f][s] + self.emission_probability[s][observation[1]]
                score[f][s] = p
                link[1][f][s] = f

        
        # 从第三时刻开始，使用前一个时刻的概率矩阵 * 二阶转移张量 * 发射概率矩阵
        pre = np.zeros((max_s, max_s))
        for i in range(2, len(observation)):
            buffer = pre
            pre = score
            score = buffer
            
            # f, s 和t 分别代表三元语法中的三个状态first, second 和 third
            for s in range(max_s):
                for t in range(max_s):
                    score[s][t] = Predefine.INTEGER_MIN_VALUE
                    for f in range(max_s):
                        p = pre[f][s] + self.transition_probability2[f][s][t] + self.emission_probability[t][observation[i]]
                        if p > score[s][t]:
                            score[s][t] = p
                            link[i][s][t] = f

        
        # 后向回溯
        max_score = Predefine.INTEGER_MIN_VALUE
        best_s, best_t = 0, 0
        for s in range(max_s):
            for t in range(max_s):
                if score[s][t] > max_score:
                    max_score = score[s][t]
                    best_s = s
                    best_t = t


        for i in range(len(link) - 1, -1, -1):
            state[i] = best_t
            best_f = link[i][best_s][best_t]
            best_t = best_s
            best_s = best_f
        
        return max_score


    def similar(self, model):
        if not isinstance(model, SecondOrderHiddenMarkovModel):
            return False
        
        print('-----')
        for i in range(len(self.transition_probability)):
            for j in range(len(self.transition_probability)):
                #print(self.transition_probability2[i][j])
                #print(model.transition_probability2[i][j])
                #print(' ')

                if not HiddenMarkovModel.static_similar(self.transition_probability2[i][j], model.transition_probability2[i][j]):
                    return False

        return super().similar(model)
