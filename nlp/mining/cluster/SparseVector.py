import math
from collections import defaultdict

"""稀疏向量"""
class SparseVector():
    def __init__(self):
        self.map = defaultdict(float)
    
    def size(self):
        return len(self.map.keys())

    def get(self, key):
        return self.map.get(key, 0.0)
    
    def put(self, key, value):
        self.map[key] = value

    def entrySet(self):
        return self.map.items()

    def clear(self):
        self.map.clear()
    
    def normalize(self):
        norm = self.norm()

        for k, v in self.entrySet():
            self.put(k, v / norm)

    def norm(self) -> float:
        """计算向量的L2范数, 也成欧几里得长度，向量元素平方和的平方根"""
        return math.sqrt(self.norm_squared())
    
    def norm_squared(self) -> float:
        _sum = 0
        for point in self.map.values():
            _sum += point * point
        return _sum

    def add_vector(self, vec):
        """这是专门用于将文档特征添加到复合向量中，特别注意这是一个累加特征值的过程"""
        for entry in vec.entrySet():
            v = self.get(entry[0])
            if not v:
                v = 0.0

            self.put(entry[0], v + entry[1])
    
    def sub_vector(self, vec):
        for entry in vec.entrySet():
            v = self.get(entry[0])
            if not v:
                v = 0.0

            self.put(entry[0], v - entry[1])

    @staticmethod
    def inner_product(vec1, vec2):
        """
        calculate the inner product value between vectors
        -param vec1 SparseVector
        -param vec2 SparseVector
        return 
        """

        it = defaultdict(float)
        
        other = None
        if vec1.size() < vec2.size():
            it = vec1.entrySet()
            other = vec2

        else:
            it = vec2.entrySet()
            other = vec1
        
        # 获取两个向量的点积
        prod = 0
        for entry in it:
            prod += entry[1] * other.get(entry[0])

        return prod
    
    def cosine(self, vec1, vec2):
        """
        calculate the cosine value between vectors
        -param vec1 SparseVector
        -param vec2 SparseVector
        """
        norm1 = vec1.norm()
        norm2 = vec2.norm()

        result = 0.0
        if norm1 == 0 and norm2 == 0:
            return result

        prod = SparseVector.inner_product(vec1, vec2)
        result = prod / (norm1 * norm2)
        
        return result

    def toString(self):
        return ','.join([str(key) +"="+ str(val) for key, val in self.map.items()])
