import sys
sys.path.append('/pynjlp')

from nlp.model.perceptron.PerceptronNameGenderClassifier import PerceptronNameGenderClassifier

"""简单的特征分类器"""
class CheapFeatureClassifier(PerceptronNameGenderClassifier):
    
    def extractFeature(self, text, featureMap):
        featureList = []
        
        # 获取姓名中的名字
        givenName = self.extractGivenName(text)

        # 特征模版: g[0], g[1] 与位置无关
        self.addFeature(givenName[0:1], featureMap, featureList)
        self.addFeature(givenName[1:], featureMap, featureList)

        return featureList

if __name__ == '__main__':
    
    base_path = '/pynjlp/data/test/cnname/'
    training_set =  base_path + 'train.csv'
    
    model = CheapFeatureClassifier()
    model.train(training_set, 10)
