from nlp.model.perceptron.model.LinearModel import LinearModel

"""对数线性模型"""
class LogLinearModel(LinearModel):
    def __init__(self, featureMap = None, parameter = None, modelFile = ""):
        """
        -param modelFile str HanLP的.bin格式，或CRF++的.txt格式（将会自动转换为model.txt.bin，下次会直接加载.txt.bin）
        """
        super().__init__(featureMap, parameter, modelFile)
