from nlp.model.perceptron.PerceptronClassifier import PerceptronClassifier

"""标准的姓名-性别感知机分类器"""
class PerceptronNameGenderClassifier(PerceptronClassifier):
    def __init__(self, model = None):
        super().__init__(model)

    def extractFeature(self, text, featureMap) -> list:
        """
        特征提取
        -param text 文本
        -param featureMap 特征映射
        return 特征向量
        """

        featureList = []
        
        givenName = self.extractGivenName(text)
        self.addFeature("1" + givenName[:1], featureMap, featureList)
        self.addFeature("2" + givenName[1:], featureMap, featureList)

        return featureList


    def extractGivenName(self, name) -> str:
        """
        提取姓名中的名字，去掉姓
        -param name 姓名
        return 名
        """
        if len(name) <= 2:
            return "_" + name[len(name) - 1:]

        return name[len(name) - 2:]



