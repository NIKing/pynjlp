class State():

    # 路径花费
    cost = 0.0

    # 当前位置
    vertex = 0

    def __init__(self, cost, vertex):
        self.cost = cost
        self.vertex = vertex

    def compareTo(self, o):
        """比较两个双精度值，"""
        return self.cost == o.cost

