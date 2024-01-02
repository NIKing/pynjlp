"""
将一个词网转为词图
"""
class Graph:
    
    # 顶点
    vertexes = []

    # 边，到达下标i
    edgesTo  = []

    def __init__(self, vertexes = []):
        self.vertexes = vertexes
        self.edgesTo  = [] * len(vertexes)

    def connect(self, _from, _to, _weight):
        """
        连接两个节点
        -param _from 起点
        -param _to 终点
        -param _weight 花费
        """
        self.edgesTo[_to].append(EdgeFrom(_from, _weight, self.vertexes[_from].word + '@' + self.vertexes[to].word))
