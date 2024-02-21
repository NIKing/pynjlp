import sys
import math

from nlp.model.hmm.HiddenMarkovModel import HiddenMarkovModel
from nlp.utility.Predefine import Predefine

"""一阶隐马尔可夫模型"""
class FirstOrderHiddenMarkovModel(HiddenMarkovModel):

    def __init__(self, start_probability = None, transition_probability = None, emission_probability = None):
        """
        -param start_probability 初始状态概率向量
        -param transition_probability 转移状态概率矩阵
        -param emission_probability 发射状态概率矩阵
        """
        super().__init__(start_probability, transition_probability, emission_probability)
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
        B  = self.logToCdf(self.emission_probability)

                
        # 定义采样长度，获取第一个显示/隐藏状态
        xy = [[0 for j in range(length)] for i in range(2)]
        xy[1][0] = self.drawFrom(pi)             # 根据初始状态概率，采样首个隐状态
        xy[0][0] = self.drawFrom(B[xy[1][0]])    # 根据发射概率矩阵，采样首个隐状态对应的首个显状态
        
        
        # 刚开始定义第一个隐状态，之后的隐状态是根据它之前隐状态（刚开始是第一个隐状态），
        # 且依据转移概率矩阵进行采样，这符合 p(y_{t+1} | y_(t)) 公式

        # 显状态并不依赖于它之前的显状态，而是仅仅依赖于该时刻的隐状态，
        # 因此下面代码中看到取 xy[1][t] 表示该时刻的隐状态，[1]表示隐状态取值，[t]表示当前时刻
        # 且依据发射概率矩阵进行采样，这符合 p(x_t | y_t) 公式
        for t in range(1, length):
            xy[1][t] = self.drawFrom(A[xy[1][t - 1]])   # 根据转移概率矩阵，继续采样，补充到状态序列
            xy[0][t] = self.drawFrom(B[xy[1][t]])   # 根据发射概率矩阵，继续采样，补充到观察序列

        return xy


    def predict(self, observation, state) -> float:
        """
        预测, 使用维特比算法，首先进行联合概率计算（前向算法），然后搜索联合概率中最大值（后向搜索），即得到结果就是最长路径
        -param observation 观测序列
        -param state 预测状态序列, 默认0，需要赋值返回
        return 概率对数
        """
        
        # 观测时序序列长度 和 状态种数
        time, max_s = len(observation), len(self.start_probability)
        
        # ----- 前向遍历计算，状态序列和观察序列的两个联合概率，因为第一时刻没有转移矩阵，就单独计算 Start ---

        # 状态得分
        score = [0.0] * max_s

        # 第一时刻，使用初始概率向量 * 发射概率矩阵
        # 注意，因为取对数的原因，所有 * 变成 +
        for i in range(max_s):
            score[i] = self.start_probability[i] + self.emission_probability[i][observation[0]]
        
        # 第二时刻，使用初始概率向量 * 一阶转移矩阵 * 发射概率矩阵计算每个字符的联合概率（这也是计算该字符的{B,M,E,S}标注集的概率），通过对比标注集概率值，得出最高值，并记录位置
        # 注意，因为取对数的原因，所有 * 变成 + 
        link, pre = [[0] * max_s for s in range(time)], [0.0] * max_s
        for t in range(1, time):
            buffer = pre
            pre = score
            score = buffer

            for s in range(max_s):
                
                score[s] = Predefine.INTEGER_MIN_VALUE 
                for f in range(max_s):

                    # [f][s] = N * N, [s][observation[t]] = N * M
                    p = pre[f] + self.transition_probability[f][s] + self.emission_probability[s][observation[t]]

                    if p > score[s]:
                        score[s] = p
                        link[t][s] = f
            
            #print(f'{t}: {score}')

        #print(link)
        # ------ 前向遍历计算 End ----------
        
        # 获取最后一个字符的最大概率标注
        max_score = Predefine.INTEGER_MIN_VALUE 
        best_s = 0
        for s in range(max_s):
            if score[s] > max_score:
                max_score = score[s]
                best_s = s
        
        # 后向回溯，找到每个最大概率标注
        for t in range(len(link) - 1, -1, -1):
            state[t] = best_s
            best_s = link[t][best_s]
        
        #print(state)
        return max_score



