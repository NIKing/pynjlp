from nlp.collection.trie.bintrie.BinTrie import BinTrie
from nlp.corpus.io.IOUtil import readlinesTxt

class Vocabulary:
    UNK = 0 # unknown
    def __init__(self, trie = None, mutable = True):
        
        if not trie:
            self.trie = BinTrie()
            self.trie.put('\t', self.UNK)
        else:
            self.trie = trie

        self.mutable = mutable # 是否只读

    def idOf(self, string):
        """字符转换词表的ID"""
        id = self.trie.get(string)

        if id == None:
            if self.mutable:
                id = self.trie.getSize()
                self.trie.put(string, id)
            else:
                id = self.UNK

        return id

    def from_pretrained(self, path):
        vocabulary = readlinesTxt(path)
        self.mutable = True
        
        self.trie = BinTrie()
        for char in vocabulary:
            self.idOf(char)

        self.mutable = False
            
    
