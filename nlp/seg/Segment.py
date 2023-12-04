
from ABC import abc, abstractmethod
from .Config import Config

class Segment(abc):
    
    config = None

    def __init__(self):
        config = Config()
    
    @abstractmethod
    def segSentence(self, sentence = str) -> list:
        pass

