import os
import sys

from nlp.seg.DictionaryBasedSegment import DictionaryBasedSegment
from nlp.seg.common.Term import Term

from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie

class DoubleArrayTrieSegment(DictionaryBasedSegment):
    
    trie = None

    def __init__(self, dictionary):
        super().__init__()

        self.trie = DoubleArrayTrie(dictionary)
        self.config.useCustomDictionary = False

    def segSentence(self, charArray = str) -> list:
        if not charArray:
            return []

        natureArray = []
        wordNet = [1] * len(charArray)
        
        # 这是用户自定义词典匹配了一次
        self.matchLongest(charArray, wordNet, natureArray, self.trie)
        
        # 这是用内置用户词典匹配，不过在这里永远等于False，除非单独设置
        if self.config.useCustomDictionary:
            self.matchLongest(charArray, wordNet, natureArray, self.customDictionary.dat)
            
            # 这是用 binTrie 还是 Trie ?
            #if self.customDictionary.trie:
            #    self.customDictionary.trie.paraseLongestText(charArray)

        termList = []
        i = 0
        
        #self.posTag(charArray, wordNet, natureArray)
        while i < len(wordNet):
            word = charArray[i: i + wordNet[i]]
            nature = ('nz' if not natureArray[i] else natureArray[i]) if self.config.speechTagging else None
            
            term = Term(word, nature)
            term.offset = i

            termList.append(term)
            i += wordNet[i]
        
        return termList

    def matchLongest(self, sentence, wordNet, natureArray, trie):
        searcher = trie.getLongestSearcher(sentence, 0)
        
        while(searcher.next()):
            #print(f'bbb--{searcher.begin}---{searcher.begin + searcher.length}')
            wordNet[searcher.begin] = searcher.length
            
            # 词性标注- value 没有 natrue 可能会引起报错
            if self.config.speechTagging:
                natureArray[searcher.begin] = searcher.value.natrue[0]
    


        

