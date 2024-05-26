from nlp.model.perceptron.tagset.CWSTagSet import CWSTagSet
from nlp.model.perceptron.instance.CWSInstance import CWSInstance

from nlp.model.perceptron.PerceptronTrainer import PerceptronTrain

class CWSTrainer(PerceptronTrain):

    def createTagSet(self):
        """创建分词标签集"""
        return CWSTagSet()

    
    def createInstance(self, sentence, mutableFeatureMap):
        """
        实例化句子
        -param sentence Sentence 对象
        -param mutableFeatureMap 特征映射对象
        """ 
        # 注意，到这里sentence是句子的对象, 要转换成单词
        #wordList = sentence.toSimpleWordList() # java的代码需要转换，python不用

        termArray = self.toWordArray(sentence)
        instance = CWSInstance(termArray, mutableFeatureMap)

        return instance
    
    def toWordArray(self, wordList) -> list:
        return [word.getValue() for word in wordList]

