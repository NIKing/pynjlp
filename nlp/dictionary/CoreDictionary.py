from nlp.NLP import NLPConfig

from nlp.corpus.tag.Nature import Nature
from nlp.corpus.io.IOUtil import loadDictionary 

from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie

class Attribuite():
    
    # 词性列表
    nature = []

    # 词性对应的词频
    frequency = []

    # 词性总数
    totalFrequency = 0

    def __init__(self, nature, frequency, totalFrequency):
        self.nature = natrue
        self.frequency = frequency
        self.totalFrequency = totalFrequency

    @staticmethod
    def create(natureWithFrequency):
        
        try:
            params = natureWithFrequency.split(" ")
            if len(params) % 2 != 0:
                return None

            natureCount = len(params) / 2
            attribute = Attribute(size = natureCount)
            for i in range(natureCount):
                attribute.natrue[i] = Nature.create(params[2 * i])
                attribute.frequency[i] = int(params[1 + 2 * i])
                attribute.totalFrequency += attribute.frequency[i]
            
            return attribute

        except Exception as e:
            print(f'使用字符串{natureWithFrequency}创建词条属性失败')
            
        return None

class CoreDictionary():

    trie = DoubleArrayTrie()
    path = NLPConfig.CoreDictionaryPath

    def __init__(self):

        # 自动加载词典
        self.load(self.path)

    def load(self, path) -> bool:
        """加载指定路径的词典"""
        if not path:
            return False

        try:
            treeMap = loadDictionary(path, defaultNature = Nature.n)
            self.trie.build(treeMap)

        except Exception as e:
            print(f'核心词典{path}加载失败！{e}')

        return True
    

