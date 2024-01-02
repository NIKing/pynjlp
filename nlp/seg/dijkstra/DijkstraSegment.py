from collections import deque

from nlp.seg.WordBasedSegment import WordBasedSegment
from nlp.seg.common.WordNet import WordNet

class DijkstraSegment(WordBasedSegment):
    """最短路径分词"""

    def segSentence(self, sentence):
        """分词"""

        # 创建词网列表
        wordNetAll = WordNet(sentence)
        #wordNetAll = WordNet(wordNetOptimum.charArray)
        
        # 生成词网
        self.generateWordNet(wordNetAll)

        # 生成词图
        graph = WordBasedSegment.generateBiGraph(wordNetAll)

        vertexList = DijkstraSegment.dijkstra(graph)

        return self.convert(vertexList, config.offset)
    
    
    @staticmethod
    def dijkstra(graph):
        """dijkstra最短路径"""
        vertexes = graph.getVertexes()
        edgesTo = graph.getEdgesTo()
        
        d = [sys.float_info.max] * len(vertexes)
        d[len(d) - 1] = 0

        path = [-1] * len(vertexes)
        
        queue = deque() 
        queue.append(State(0, len(vertexes) - 1))

        while len(queue) > 0:
            p = queue.popleft()
            if d[p.vertex] < p.cost:
                continue

            for edgeFrom in edgesTo[p.vertex]:
                if d[edgeFrom._from] > d[p.vertex] + edgeFrom.weight:
                    d[edgeFrom._from] = d[p.vertex] + edgeFrom.weight
                    queue.append(State(d[edgeFrom._from], edgeFrom._from))

                    path[edgeFrom._from] = p.vertex
        
        resultList = []
        t = 0
        while t != -1:
            resultList.append(vertexes[t])
            t = path[t]

        return resultList
