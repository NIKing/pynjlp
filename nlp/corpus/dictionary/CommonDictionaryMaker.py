from abc import ABC, abstractmethod

from nlp.corpus.dictionary.DictionaryMaker import DictionaryMaker
from nlp.corpus.dictionary.NGramDictionaryMaker import NGramDictionaryMaker

class CommonDictionaryMaker(ABC):

    def __init__(self):
        self.dictionaryMaker = DictionaryMaker()
        self.nGramDictionaryMaker = NGramDictionaryMaker()

    @abstractmethod
    def roleTag(self, sentence_list):
        """角色标注，如果要增加新的label或增加首尾可在此进行"""
        pass
    
    @abstractmethod
    def addDictionary(self, sentence_list):
        """添加到词典，比较灵活的添加，不用直接放在字典中"""
        pass

    def compute(self, sentenceList):
        self.sentenceList = sentenceList

        self.roleTag()
        self.addDictionary()
    
    def saveTxtTo(self, path):
        """保存普通词典和n元语法词典"""
        self.dictionaryMaker.saveTxtTo(path + '.txt')
        self.nGramDictionaryMaker.saveTxtTo(path)
