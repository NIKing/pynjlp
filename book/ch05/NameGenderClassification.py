import sys
sys.path.append('/pynjlp')

from nlp.model.perceptron.PerceptronNameGenderClassifier import PerceptronNameGenderClassifier

"""简单的特征分类器"""
class CheapFeatureClassifier(PerceptronNameGenderClassifier):
    def extractFeature(self, text, featureMap):
        """特征提取"""
        featureList = []
        
        # 特征函数，获取姓名中的名字
        givenName = self.extractGivenName(text)
        
        # 特征模版，g[0], g[1] 与位置无关
        self.addFeature(givenName[:1], featureMap, featureList)
        self.addFeature(givenName[1:], featureMap, featureList)

        return featureList

"""复杂的特征分类器"""
class RichFeatureClassifier(PerceptronNameGenderClassifier):
    def extractFeature(self, text, featureMap):
        """特征提取"""
        featureList = []
        
        # 特征函数
        givenName = self.extranctGivenName(text)

        # 特征模版
        self.addFeature("1" + givenName[:1], featureMap, featureList)
        self.addFeature("2" + givenName[1:], featureMap, featureList)
        self.addFeature("3" + givenName, featureMap, featureList)

        return featureList

def test_name(classifier):
    for name in ['赵建军', '沈雁冰', '陆雪琪', '李冰冰', '倪杰', '李哲']:
        res = classifier.predict(name)
        print('%s=%s' % (name, res))

def trainAndEvaluate(template, classifier, averagePerceptron):
    base_path = '/pynjlp/data/test/cnname/'
    training_set =  base_path + 'train_small.csv'

    #training_set =  base_path + 'train.csv'
    testing_set  =  base_path + 'test.csv'
    model_path   =  base_path + 'cnname.bin'
    
    accuracy = classifier.train(training_set, 10, averagePerceptron)
    print(f'训练集准确率：{accuracy}')

    model = classifier.getModel()
    print(f'特征数量：{len(model.parameter)}')
    
    #test_accuracy = classifier.evaluate(testing_set)
    #print(f'测试集准确率：{test_accuracy}')

    model.save(model_path, None, 0, True)

    #test_name(classifier)

if __name__ == '__main__':
    
    trainAndEvaluate("简单的特征模版", CheapFeatureClassifier(), False)
    #trainAndEvaluate("简单的特征模版", CheapFeatureClassifier(), True)
    
    #trainAndEvaluate("标准的特征模版", PerceptronNameGenderClassifier(), False)
    #trainAndEvaluate("标准的特征模版", PerceptronNameGenderClassifier(), True)

    #trainAndEvaluate("复杂的特征模版", RichFeatureClassifier(), False)
    #trainAndEvaluate("复杂的特征模版", RichFeatureClassifier(, True)
