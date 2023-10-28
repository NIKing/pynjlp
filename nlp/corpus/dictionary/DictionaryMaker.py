
"""
一个通用的词典制作工具，词条格式：词 标签 词频
"""
import sys
sys.path.append('pynjlp')

from nlp.collection.trie.bintrie.BinTrie import BinTrie

class DictionaryMaker():

    def __init__(self):
        self.trie = BinTrie()
