from abc import ABC, abstractmethod

from nlp.corpus.dictionary.DictionaryMaker import DictionaryMaker
from nlp.corpus.dictionary.NGramDictionaryMaker import NGramDictionaryMaker

class CommonDictionaryMaker(ABC):

    def __init__(self):
        self.dictionaryMaker = DictionaryMaker()
        self.nGramDictionaryMaker = NGramDictionaryMaker()

    
    @abstractmethod
    def roleTag(self, sentece_list):
        """角色标注，如果要增加新的label或增加首尾可在此进行"""
        pass
    
    @abstractmethod
    def addDictionary(self, sentece_list):
        """添加到词典，比较灵活的添加，不用直接放在字典中"""
        pass

    def compute(self, sentence_list):
        self.roleTag(sentece_list)
        self.addDictionary(sentence_list)
    
    def saveTo(self, path):
        self.dictionaryMaker.saveTo(path)
        self.nGramDictionaryMaker.saveTo(path)
