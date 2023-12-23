"""
 * 使用AhoCorasickDoubleArrayTrie实现的最长分词器<br>
 * 需要用户调用setTrie()提供一个AhoCorasickDoubleArrayTrie
"""
from nlp.seg.DictionaryBasedSegment import DictionaryBasedSegment
from nlp.seg.common.Term import Term

from nlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie import AhoCorasickDoubleArrayTrie

class AhoCorasickDoubleArrayTrieSegment(DictionaryBaseSegment):
    
    trie = None

    def __init__(self, dictionaryPaths):
        super().__init__()
        
        self.trie = AhoCorasickDoubleArrayTrie()

        self.config.useCustomDictionary = False
        self.config.speechTagging = False

        if dictionaryPaths:
            self.loadDictionary(dictionaryPaths)

    def setTrie(self, trie):
        if not trie:
            return

        self.trie = trie

    def getTrie(self):
        return self.trie

    def loadDictionary(self, pathArray):
        if not pathArray or len(pathArray) <= 0:
            return
    
        treeMap = {}
        try:
            for path in pathArray:
                treeMap.update(loadDictionary(pathArray))
        except Exception as e:
            print(f'加载字典失败：{e}')

        if treeMap and len(treeMap.keys()) > 0:
            self.trie.build(treeMap)
            
        return self


    def segSentence(self, charArray = str) -> list:
        if not charArray:
            return []
        
        natureArray = []
        wordNet = [1] * len(charArray)

        searcher = self.trie.getSearcher(charArray, 0)
        while(searcher.next()):
            #print(f'bbb--{searcher.begin}---{searcher.begin + searcher.length}')
            wordNet[searcher.begin] = searcher.length
            
            # 词性标注- value 没有 natrue 可能会引起报错
            if self.config.speechTagging:
                natureArray[searcher.begin] = searcher.value.natrue[0]
    

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


