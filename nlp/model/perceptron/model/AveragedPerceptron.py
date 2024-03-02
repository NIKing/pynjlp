from nlp.model.perceptron.model.LinearModel import LinearModel

class AveragedPerceptron(LinearModel):
    def __init__(self, featureMap = None, parameter = None):
        super().__init__(featureMap, parameter)

    def update(self, index, value, total, timestamp, current):
        """
        根据答案和预测更新参数
        -param index        特征向量的下标/特征向量/预测正确的特征函数
        -param value        更新量/命中的特征函数
        -param total        权值向量总和
        -param timestamp    每个权值上次更新的时间戳
        -param current      当前时间戳
        """

        # 当index = 预测正确的特征函数；value = 命中的特征函数
        if isinstance(index, list) and isinstance(value, list):
            for i in range(len(index)):
                if index[i] == value[i]:
                    continue
                else:
                    self.update(index[i], 1, total, timestamp, current)
                    if value[i] >= 0 and value[i] < len(self.parameter):
                        self.update(value[i], -1, total, timestamp, current)
                    else:
                        raise ValueError('更新参数时传入了非法下标')
        
        # 当index = 特征向量
        elif isinstance(index, list):
            for i in index:
                self.update(i, value, total, timestamp, current)
        
        # 当index = 特征向量的下标
        else:
            passed = current - timestamp[index]
            total[index] += passed * self.parameter[index]
            self.parameter[index] += value
            timestamp[index] = current

    def average(self, total, timestamp, current):
        for i in range(len(self.parameter)):
            self.parameter[i] = (total[i] + (current - timestamp[i]) * self.parameter[i]) / current


