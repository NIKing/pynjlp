from nlp.model.perceptron.model.LinearModel import LinearModel
from nlp.model.perceptron.instance.CWSInstance import CWSInstance

from nlp.model.perceptron.PerceptronTagger import PerceptronTagger
from nlp.dictionary.other.CharTable import CharTable

class PerceptronSegmenter(PerceptronTagger):
    def __init__(self, cwsModelFile = ""):
        
        cwsModel = self.initModel(cwsModelFile)
        super().__init__(cwsModel)
        
        self.CWSTagSet = None
        if cwsModel != None:
            self.CWSTagSet = cwsModel.featureMap.tagSet

    def initModel(self, cwsModelFile = ""):
        cwsModel = None
        if isinstance(cwsModelFile, str) and cwsModelFile != "":
            cwsModel = LinearModel(modelFile = cwsModelFile)
        elif cwsModelFile:
            cwsModel = cwsModelFile

        return cwsModel

    def segment(self, sentence, normalized = "", instance = None, output = []):
        if not sentence:
            return 
        
        # 实例化当前句子，会创建当前句子的特征向量
        if not normalized and not isinstance(normalized, str):
            normalized = CharTable.normalize(sentence)

        if not instance and normalized:
            instance = CWSInstance(normalized, self.model.featureMap)
        
        # 当前句子的标准标签集, 这里怎么和 StructurePerceptron 不太一样，viterbiDecode() 第二个参数不是guessLabel吗？
        tagArray = instance.tagArray
        self.model.viterbiDecode(instance, tagArray)

        temp_result = [sentence[0]]        
        for i in range(1, len(tagArray)):
            if tagArray[i] == self.CWSTagSet.B or tagArray[i] == self.CWSTagSet.S:
                output.append(''.join(temp_result))
                temp_result = []
            
            temp_result.append(sentence[i])

        if len(temp_result) > 0:
            output.append(''.join(temp_result))

    def createInstance(self, sentence, featureMap):
        return CWSInstance.create(sentence, featureMap)
    
    def get_char_feature(self, sentence, char):
        if not char:
            return ''

        instance = CWSInstance(sentence, self.model.featureMap)
        char_position = sentence.find(char)

        if char_position == -1:
            return ''
        
        char_feature_vector = instance.getFeatureAt(char_position)

        print(self.model.featureMap.tagSet.size())
        
        # 根据特征向量的去权重向量中取不同标签的权重
        char_feature_matrix = [[] for i in char_feature_vector]
        for i, index in enumerate(char_feature_vector):
            
            print('')
            print(index)
            feature_vector = []
            for j, tag in enumerate(self.model.featureMap.allLabels()):
                
                if index < -1 or index >= self.model.featureMap.getSize():
                    raise ValueError('非法下标')

                index = index * self.model.featureMap.tagSet.size() + j
                print(' ', index)

                feature_vector.append(self.model.parameter[index])
            
            char_feature_matrix[i] = feature_vector
            
        print(instance.tagArray)
        print(instance.featureMatrix)

        print(char_feature_matrix)



