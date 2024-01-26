
"""
也是最短路分词，最短路求解采用Viterbi算法
"""
from nlp.seg.WordBasedSegment import WordBasedSegment
from nlp.seg.Config import Config

from nlp.seg.common.WordNet import WordNet

class ViterbiSegment(WordBasedSegment):
    
    def __init__(self):
        super().__init__()

    def segSentence(self, sentence):
        """分词"""

        # 生成词网
        wordNetAll = WordNet(sentence)

        self.generateWordNet(wordNetAll)

        vertexList = ViterbiSegment.viterbi(wordNetAll)

        return self.convert(vertexList, Config.offset)
    
    @staticmethod
    def viterbi(wordNet) -> list:
        """维比特算法"""
        nodes = wordNet.getVertexes()
        vertexList = []
        
        #print(nodes[0][0].toString())
        #print(nodes[1][0].toString())
        #print(nodes[1][0]._from)
        #print(nodes[4][0]._from)

        # 前向遍历： 由起点出发向后遍历，更新从起点到该节点的最小花费以及前驱指针
        for node in nodes[1]:
            node.updateFrom(nodes[0].pop(0))

        for i in range(1, len(nodes) - 1):
            nodeArray = nodes[i]
            if not nodeArray:
                continue

            for node in nodeArray:
                if not node._from:
                    continue
                
                # 根据行索引构建词图的良好性质，更新每个节点的前驱节点
                # 根据距离公式计算节点距离(在updateFrom中计算距离)，并维护最短路径上的前驱指针from
                for to in nodes[i + len(node.realWord)]:
                    to.updateFrom(node)

        # 后向回溯：由终点出发从后往前回溯前驱指针，取得最短路径
        _from = nodes[len(nodes) - 1].pop(0)
        while _from:
            vertexList.append(_from)
            _from = _from._from

        vertexList.reverse()

        return vertexList

