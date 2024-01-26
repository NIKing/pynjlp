from nlp.collection.trie.bintrie.BinTrie import BinTrie
from nlp.corpus.io.IOUtil import writeTxtByList 

from nlp.corpus.dictionary.TMDictionaryMaker import TMDictionaryMaker

"""2-gram词典制作工具"""
class NGramDictionaryMaker():

    def __init__(self):
        self.trie = BinTrie()
        self.tmDictionaryMaker = TMDictionaryMaker()

    def addPair(self, first, second):
        combine = first.getValue() + "@" + second.getValue()
        frequency = self.trie.get(combine)
        if not frequency:
            frequency = 0

        self.trie.put(combine, frequency + 1)
        #print(f'combine==={combine}')

        #同时统计标签转移情况 - 统计两个单词组合的次数
        self.tmDictionaryMaker.addPair(first.getLabel(), second.getLabel())

    def saveTxtTo(self, path):
        """保存NGram词典和转移矩阵"""
        self.saveNGramToTxt(path + '.ngram.txt')
        self.saveTransformMatrixToTxt(path + '.tr.txt')

    def saveNGramToTxt(self, path):
        """保存NGram词典"""
        try:
            entries = self.trie.entrySet()
            entries = [key +"\t"+ str(value) for key, value in entries.items()]
            
            writeTxtByList(path, entries)
        except Exception as e:
            print(f'在保存NGram词典到【{path}】时发生异常【{e}】')
            return False

        return True
    
    def saveTransformMatrixToTxt(self, path):
        """保存转移矩阵"""
        self.tmDictionaryMaker.saveTxtTo(path)

