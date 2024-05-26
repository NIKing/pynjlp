from abc import ABC, abstractmethod
from nlp.seg.Segment import Segment

class CharacterBasedSegment(Segment, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def segSentence(self, sentence):
        if len(sentence) == 0:
            return []

        termList = self.roughSegSentence(sentence)
        if not self.config.ner or self.config.userCustomDictionary or self.config.speechTagging:
            return termList

        vertexList = self.toVertextList(termList, True)
        if self.config.speechTagging:
            Viterbi.compute(vertexList, CoreDictionaryTransformMatrixDictionary.transformMatrixDictionary)
            i = 0
            for term in termList:
                if not term.nature:
                    term.nature = vertexList.get(i + 1).guessNature()
                ++i
        
        if self.config.useCustomDictionary:
            self.combineByCustomDictionary(vertexList)
            termList = self.convert(vertexList, self.config.offset)

        return termList

    @abstractmethod
    def roughSegSentence(self, sentence):
        """单纯的分词模型"""
        pass

