from nlp.NLP import NLPConfig

from nlp.corpus.tag.Nature import Nature
from nlp.corpus.io.IOUtil import loadDictionary 

from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie

class Attribute():
    
    # 词性列表
    nature = []

    # 词性对应的词频
    frequency = []

    # 词性总数
    totalFrequency = 0

    def __init__(self, nature, frequency, totalFrequency = 0):
        if not isinstance(nature, list):
            self.nature.append(nature)
            self.frequency.append(frequency)
            self.totalFrequency = frequency
        else:
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
    """使用DoubleArrayTrie实现的核心词典"""

    trie = DoubleArrayTrie()

    path = NLPConfig.CoreDictionaryPath
    
    def __init__(self):
        self.load(self.path)

    def load(self, path) -> bool:
        """加载指定路径的词典"""
        if not path:
            return False

        try:
            treeMap = loadDictionary(path, splitter = ' ',  defaultNature = Nature.n)
            self.trie.build(treeMap)
            print(f'核心词典 {path} 加载成功')
        except Exception as e:
            print(f'核心词典 {path} 加载失败！{e}')

        return True

    def reload(self):
        path = NLPConfig.CoreDictionaryPath
        return self.load(path)
    
    @staticmethod
    def getWordID(a):
        """
        获取词语的ID
        -param a 词语
        return ID，如果不存在，则返回 -1
        """
        return CoreDictionary.trie.exactMatchSearch(a)

