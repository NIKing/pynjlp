import sys
sys.path.append('/pynjlp')

from nlp.collection.trie.bintrie.BinTrie import BinTrie
from nlp.corpus.io.IOUtil import writeTxtByList 

"""2-gram词典制作工具"""
class NGramDictionaryMaker():

    def __init__(self):
        self.trie = BinTrie()
        #self.tmDictionaryMaker = TMDictionaryMaker()

    def addPair(self, first, second):
        combine = first.getValue() + "@" + second.getValue()

        frequency = self.trie.get(combine)
        if not frequency:
            frequency = 0

        self.trie.put(combine, frequency+1)

        #同时统计标签转移情况
        #self.tmDictionaryMaker.addPair(first.getLabel(), second.getLabel())

    def saveTxtTo(self, path):
        """保存NGram词典和转移矩阵"""
        self.saveNGramToTxt(path)
        #self.saveTransformMatrixToTxt()

    def saveNGramToTxt(self, path):
        """保存NGram词典"""
        try:
            entries = self.trie.entrySet()
            print(entries)

            entries = [key +" "+ value for key, value in entries.items()]
            
            print(path, entries)
            writeTxtByList(path, entries)
        except Exception as e:
            print(f'保存到【{path}】失败【{e}】')
            return False

        return True
    
    def saveTransformMatrixToTxt(self, path):
        """保存转移矩阵"""
        self.tmDictionaryMaker.saveTxtTo(path)

