from collections import defaultdict
from nlp.corpus.io.IOUtil import writeTxtByList 

"""转移矩阵词典制作工具"""
class TMDictionaryMaker:

    def __init__(self):
        self.transferMatrix = defaultdict(int)
    
    def addPair(self, first, second):
        """
        添加一个转移例子，会在内部统计完成
        """
        firstMatrix = self.transferMatrix.get(first)
        if not firstMatrix:
            firstMatrix = defaultdict(int)
            self.transferMatrix[first] = firstMatrix

        frequency = firstMatrix.get(second)
        if not frequency:
            frequency = 0
        
        firstMatrix[second] = frequency + 1
    
    def toString(self):
        """返回矩阵格式形式的字符串"""
        labelSet = []
        for key, val in self.transferMatrix.items():
            labelSet.append(key)
            labelSet.extend([k for k in val.keys()])

        #print(labelSet)
        matrix, items = [], [' ']
        for key in labelSet:
            items.append(key)
        matrix.append(','.join(items))

        for first in labelSet:
            firstMatrix = self.transferMatrix.get(first)
            if not firstMatrix:
                firstMatrix = defaultdict(int)

            line = [first]
            for second in labelSet:
                frequency = firstMatrix.get(second)
                if not frequency:
                    frequency = 0

                line.append(str(frequency))

            matrix.append(','.join(line))
         
        return matrix

    def saveTxtTo(self, path):
        try:
            writeTxtByList(path, self.toString())
        except Exception as e:
            print(f'保存转移矩阵词典到【{path}】时发生异常【{e}】')
            return False

        return True
