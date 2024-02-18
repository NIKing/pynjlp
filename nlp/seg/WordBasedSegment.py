from nlp.seg.Segment import Segment
from nlp.seg.common.Vertex import Vertex

from nlp.dictionary.CoreDictionary import CoreDictionary, Attribute

class WordBasedSegment(Segment):
    """基于词语NGram模型的分词器基类"""

    def __init__(self):
        super().__init__()
    
    def generateWordNet(self, wordNetStorage):
        """生成一元词网, 根据一元模型生成词网，根据对句子进行全切分生存"""

        # 通过字典树切分句子，得到句子中所有单词 —完全切分
        charArray = wordNetStorage.charArray # 这已经把句子转化成一个一个单词列表了
        searcher = CoreDictionary.trie.getSearcher(charArray, 0)
        while searcher.next():
            # 从 +1 开始，因为词网前后是开始和结尾标记
            word = ''.join(charArray[searcher.begin : searcher.begin + searcher.length])
            wordNetStorage.add(searcher.begin + 1, Vertex(" ", word, Attribute(**searcher.value), searcher.index))
        
        #print(charArray)
        #print(wordNetStorage.toString())

        # 原子分词，保证图连通
        vertexes = wordNetStorage.getVertexes()
        i = 0
        while i < len(vertexes):
            # 空白行
            if len(vertexes[i]) == 0:

                # 寻找第一个非空白行
                j = i + 1
                for j in range(i + 1, len(vertexes) - 1):
                    if len(vertexes[j]):
                        break
                
                # 填充[i, j]之间的空白行
                wordNetStorage.addAtom(i, self.quickAtomSegment(charArray, i - 1, j - 1))
                i = j
            else:
                i += len(vertexes[i][-1].realWord)

    @staticmethod
    def generateBiGraph(wordNet):
        """根据一元词网，通过一定规则生成二元词图"""
        return wordNet.toGraph()

