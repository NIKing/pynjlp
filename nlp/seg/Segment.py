from abc import ABC, abstractmethod

from nlp.seg.Config import Config
from nlp.dictionary.CustomDictionary import CustomDictionary

class Segment(ABC):
    
    config = None

    customDictionary = CustomDictionary.DEFAULT

    def __init__(self):
        self.config = Config()
    
    @abstractmethod
    def segSentence(self, sentence = str) -> list:
        pass
    
    def seg(self, sentence = str) -> list:
        return [term.toString() for term in self.segSentence(sentence)]

    def enableCustomDictionary(self, customDictionary):
        self.config.useCustomDictionary = True
        self.customDictionary = customDictionary
        
        return self
