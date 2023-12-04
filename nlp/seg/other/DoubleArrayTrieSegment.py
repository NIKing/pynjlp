import os
import sys

#root_path = os.

from nlp.seg.Segment import Segment
from nlp.collection.trie.DoubleArrayTrie

class DoubleArrayTrieSegment(Segment):
    
    trie = None

    def __init__(self):
        Segment.__init__(self)

        self.trie = DoubleArrayTrie()

    def segSentence(self, sentence = str) -> list:
        
        charArray = sentence
        natureArray = []
        wordNet = [0] * len(sentence)

        

