from nlp.model.hmm.HMMTrainer import HMMTrainer
from nlp.model.perceptron.tagset.CWSTagSet import CWSTagSet

from nlp.dictionary.other.CharTable import CharTable

#from pyhanlp import *
#CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')

"""模型分词器"""
class HMMSegmenter(HMMTrainer):

    def __init__(self, model, vocabulary = None):
        super().__init__(model, vocabulary)

        self.tagSet = CWSTagSet()
    
    def getTagSet(self):
        return self.tagSet

    def convertToSequence(self, sentence):
        """转换句子为{B,M,E,S}标签"""
        charList = []
        
        for w in sentence.wordList:
            word = CharTable.convert(w.value)   # 正规化字符，大小写转换，繁简体转换
            
            # 单字成词
            if len(word) == 1:
                charList.append((word, "S"))
            else:
                # 整个词标记：词首，词中和词尾
                charList.append((word[0:1], "B"))

                for i in range(1, len(word) - 1):
                    charList.append((word[i:i + 1], "M"))
                
                charList.append((word[len(word) - 1:], "E"))
        
        return charList


    def segment(self, origin_txt) -> list:
        # 字符正规化
        txt = CharTable.convert(origin_txt)

        # 被预测的句子，词表转换编码
        obsArray = [self.vocabulary.idOf(t) for t in txt]
        
        # 进行模型预测，tagArray = 结果
        tagArray = [0] * len(txt)
        self.model.predict(obsArray, tagArray)
        
        #print(self.tagSet.stringIdMap.keys())
        #print([(tag, self.tagSet.idStringMap[tag]) for tag in tagArray])

        # 根据标注进行截断
        wordList, result = [], [origin_txt[0]]
        for i in range(1, len(tagArray)):
            
            if tagArray[i] == self.tagSet.B or tagArray[i] == self.tagSet.S:
                wordList.append(''.join(result))
                result = []
            
            result.append(origin_txt[i])

        if len(result) != 0:
            wordList.append(''.join(result))

        return wordList

    def seg(self, txt = str) -> list:
        return self.segment(txt)






