
class LinearModel():

    def __init__(self, featureMap = None, parameter = None, modelFile = ""):
        """
        初始化
        -param featureMap 特征函数
        -param parameter 特征权重
        -modelFile 模型路径
        """

        self.featureMap = featureMap
        self.parameter = parameter
        
        if modelFile != "":
            self.load(modelFile)

    def decode(self, x):
        """
        分离超平面解码
        -param x 特征向量
        return sign(x)
        """
        y = 0
        for f in x:
            y += self.parameter[f]

        return -1 if y < 0 else 1

    def update(self, x, y):
        """
        参数更新
        -param x 特征向量
        -param y 正确答案
        """
        assert y == 1 or y == -1 , "感知机标签y必须是±1"

        for f in x:
            self.parameter[f] += y




