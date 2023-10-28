
from trie.bintrie.BinTrie import BinTrie

class Vocabulary:
    
    UNK = 0

    def __init__(self, mutable):
        self.trie = BinTrie()
        self.mutable = mutable # 是否可变
        
        self.trie.put('\t', UNK)

    def idOf(self, string):

        id = self.trie.get(string)
        
        if id == None:

            if self.mutable:
                id = self.trie.size()
                self.trie.put(string, id)
            else:
                id = UNK

        return id 
            
    
