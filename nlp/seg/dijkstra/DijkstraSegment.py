from collections import deque

from nlp.seg.WordBasedSegment import WordBasedSegment
from nlp.seg.Config import Config

from nlp.seg.common.WordNet import WordNet
from nlp.seg.dijkstra.path.State import State

import sys

class DijkstraSegment(WordBasedSegment):
    """最短路径分词"""

    def segSentence(self, sentence):
        """分词"""

        # 创建词网
        wordNetAll = WordNet(sentence)
        
        # 一元语法模型，生成词网
        self.generateWordNet(wordNetAll)
        #print(wordNetAll.toString())

        # 生成词图
        graph = WordBasedSegment.generateBiGraph(wordNetAll)
        #print(graph.toString())
        
        # 二元模型解码任务，就是在词图上找出最理想的路径
        vertexList = DijkstraSegment.dijkstra(graph)

        return self.convert(vertexList, Config.offset)
    
    
    @staticmethod
    def dijkstra(graph):
        """dijkstra 最短路径算法"""
        vertexes = graph.getVertexes()
        edgesTo = graph.getEdgesTo()
        
        print(graph.toString())

        d = [sys.float_info.max] * len(vertexes)
        d[len(d) - 1] = 0
        
        # 存储路径
        path = [-1] * len(vertexes)
        
        queue = deque() 
        queue.append(State(0, len(vertexes) - 1))
        
        # 找到最短路径放在 path 中
        while len(queue) > 0:
            p = queue.popleft()

            # 对比来自同一个方向(from)节点权重，保留权重值比较小的节点
            # 意思是说，同一个节点有两个边，保留边权重值小的
            if d[p.vertex] < p.cost:
                continue
            
            # 从最后面节点向前开始找，找到每个边
            for edgeFrom in edgesTo[p.vertex]:

                # 注意，最后一个节点在 d 中的值为 0，即刚开始 d[p.vertex] = 0，
                # 这句就是，把 d 里面值逐渐替换成节点的权重值, 并且填充路径数组（每条路径的位置上补充指向的顶点索引）
                if d[edgeFrom._from] > d[p.vertex] + edgeFrom.weight:
                    d[edgeFrom._from] = d[p.vertex] + edgeFrom.weight
                    queue.append(State(d[edgeFrom._from], edgeFrom._from))

                    path[edgeFrom._from] = p.vertex
        
        print(d)
        print(path)

        # 根据path存储的索引，获取结果
        resultList = []
        t = 0
        while t != -1:
            resultList.append(vertexes[t])
            t = path[t]
        
        return resultList
