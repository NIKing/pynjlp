"""
将一个词网转为词图
"""
from nlp.seg.common.EdgeFrom import EdgeFrom

class Graph:
    
    # 顶点
    vertexes = []

    # 边，到达下标i
    edgesTo  = []

    def __init__(self, vertexes = []):
        self.vertexes = vertexes
        self.edgesTo  = [[] for i in range(len(vertexes))]

    def connect(self, _from, _to, _weight):
        """
        连接两个节点
        -param _from 起点
        -param _to 终点
        -param _weight 花费
        """
        #print(_from, _to)
        #print(self.vertexes[_from].toString())
        #print(self.vertexes[_to].toString())

        self.edgesTo[_to].append(EdgeFrom(_from, _weight, self.vertexes[_from].toString() + '@' + self.vertexes[_to].toString()))
        
        #print(self.edgesTo)
        #print(' ')

    def getVertexes(self):
        return self.vertexes

    def getEdgesTo(self):
        return self.edgesTo

    def toString(self):

        _str = "Graph {"
        _str += "\n"
        _str += "vertexes =【" + '-'.join([vertex.toString() for vertex in self.vertexes]) + "】"
        _str += "\n"
        _str += "edgesTo =【" + '-'.join([edge.toString() for edges in self.edgesTo for edge in edges]) + "】"
        _str += "}"

        return _str
