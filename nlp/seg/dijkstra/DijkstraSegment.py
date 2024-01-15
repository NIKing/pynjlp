from collections import deque

from nlp.seg.WordBasedSegment import WordBasedSegment
from nlp.seg.common.WordNet import WordNet
from nlp.seg.dijkstra.path.State import State
from nlp.seg.Config import Config

import sys

class DijkstraSegment(WordBasedSegment):
    """最短路径分词"""

    def segSentence(self, sentence):
        """分词"""

        # 创建词网
        wordNetAll = WordNet(sentence)
        
        # 一元语法模型，生成词网
        self.generateWordNet(wordNetAll)

        # 生成词图
        graph = WordBasedSegment.generateBiGraph(wordNetAll)
        
        # 二元模型解码任务，就是在词图上找出最理想的路径
        vertexList = DijkstraSegment.dijkstra(graph)

        return self.convert(vertexList, Config.offset)
    
    
    @staticmethod
    def dijkstra(graph):
        """dijkstra 最短路径算法"""
        vertexes = graph.getVertexes()
        edgesTo = graph.getEdgesTo()
        
        d = [sys.float_info.max] * len(vertexes)
        d[len(d) - 1] = 0
        
        # 存储路径
        path = [-1] * len(vertexes)
        
        queue = deque() 
        queue.append(State(0, len(vertexes) - 1))
        
        # 找到最短路径放在 path 中
        while len(queue) > 0:
            p = queue.popleft()
            if d[p.vertex] < p.cost:
                continue
            
            # 对比当前节点后面的每个边的权重，从最后面节点向前找
            for edgeFrom in edgesTo[p.vertex]:

                # 注意，最后一个节点在 d 中的值为 0，即刚开始 d[p.vertex] = 0，后面逐渐被替换成节点的权重值
                if d[edgeFrom._from] > d[p.vertex] + edgeFrom.weight:
                    d[edgeFrom._from] = d[p.vertex] + edgeFrom.weight
                    queue.append(State(d[edgeFrom._from], edgeFrom._from))

                    path[edgeFrom._from] = p.vertex
        
        # 根据path存储的索引，获取结果
        resultList = []
        t = 0
        while t != -1:
            resultList.append(vertexes[t])
            t = path[t]
        
        return resultList
