from nlp.seg.viterbi.ViterbiSegment import ViterbiSegment
from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger

from nlp.dictionary.CoreBiGramTableDictionary import CoreBiGramTableDictionary
from nlp.dictionary.stopword.CoreStopWordDictionary import CoreStopWordDictionary

from nlp.mining.cluster.SparseVector import SparseVector
from nlp.mining.cluster.ClusterAnalyzer import ClusterAnalyzer
from nlp.mining.cluster.Document import Document

from collections import defaultdict

CoreBiGramTableDictionary.reload()
CoreStopWordDictionary.reload()

class DocumentClusterAnalyzer(ClusterAnalyzer):
    
    def __init__(self):
        super().__init__()

        self.segment = ViterbiSegment()
        self.vocabulary = MutableDoubleArrayTrieInteger()
    
    def addDocument(self, id, document):
        if not document:
            return []

        if isinstance(document, str):
            document = self.preprocess(document)
        
        # 句子分词后，转换为向量，再转换为文档
        vector = self.toVector(document)
        d = Document(id, vector)
        
        self.documents[id] = d

    def preprocess(self, document) -> list:
        """预处理、分词、去除停用词"""
        termList = self.segment.segSentence(document)

        wordList = []
        for term in termList:
            if CoreStopWordDictionary.contains(term.word) or term.nature.startswith("w"):
                continue
           
            wordList.append(term.word)
        
        return wordList

    def id(self, word):
        """建立分词索引，基于双数组字典树"""
        id = self.vocabulary.get(word)
        if id == -1:
            id = self.vocabulary.getSize()
            self.vocabulary.put(word, id)

        return id

    def toVector(self, wordList):
        """将文档转换向量，以双数组字典树为基础建立特征索引，以词频建立特征值"""
        vector = SparseVector()

        for word in wordList:
            id = self.id(word)
            f  = vector.get(id)
            
            if f == 0.0:
                f = 1.0
                vector.put(id, f)
            else:
                vector.put(id, ++f)

        return vector
