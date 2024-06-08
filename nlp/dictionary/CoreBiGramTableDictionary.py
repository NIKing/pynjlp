from nlp.NLP import NLPConfig

from nlp.corpus.tag.Nature import Nature
from nlp.corpus.io.IOUtil import loadDictionary, readlinesTxt

from nlp.dictionary.CoreDictionary import CoreDictionary

class CoreBiGramTableDictionary():
    """核心词典的二元接续词典，采用整型储存，高性能"""
    coreDictionary = CoreDictionary() 

    # 描述了词在pair中的范围，给定一个词idA，从pair[start[idA]]开始的start[idA + 1] - start[idA]描述了一些接续的频次
    start = []
    
    # pair[偶数n]表示key，pair[n+1]表示frequency
    pair = []
    
    path = NLPConfig.BiGramDictionaryPath

    def __init__(self):
        self.load(self.path)
    
    @staticmethod
    def load(path):
        self = CoreBiGramTableDictionary

        treeMap = {}
        try:
            lines = readlinesTxt(path)
            
            total = 0
            maxWordId = self.coreDictionary.trie.getSize()

            for line in lines:
                params = line.rstrip('\n').split('\t')
                towWord = params[0].split('@')
                
                # 获取整个单词在双数组字典中的base的值(begin)
                a = towWord[0]
                idA = self.coreDictionary.trie.exactMatchSearch(a)
                if idA == -1:
                    continue
                
                b = towWord[1]
                idB = self.coreDictionary.trie.exactMatchSearch(b)
                if idB == -1:
                    continue

                #print(a, idA)
                #print(b, idB)
                #print("")
                
                # biMap 映射第二单词与频次的关系
                # treeMap 映射第一个单词与biMap的关系
                biMap = treeMap.get(idA)
                if not biMap:
                    biMap = {}

                freq = int(params[1])
                biMap[idB] = freq
                treeMap[idA] = biMap

                total += 2
            
            #print(treeMap)
            # pair 以数组方式存储 treeMap.values()，用左右相邻位置存储单词的双数组中的base的值 和（第二单词和词频的映射关系表）
            # start 以字典方式存储第一个单词在双数组的 begin 值以及 treeMap 的数量。
            self.pair  = [0] * total
            self.start = [0] * (maxWordId + 1)
            
            offset = 0
            for i in range(maxWordId):
                bMap = treeMap.get(i)
                if bMap:
                    for key, value in bMap.items():
                        index = offset * 2 

                        self.pair[index] = key
                        self.pair[index + 1] = value

                        offset += 1
                
                # 为了下面计算长度 
                self.start[i + 1] = offset
            
            #print(self.pair)
            #print(self.start[:15])
            print(f'二元词典 {path} 构建成功')
            
        except Exception as e:
            print(f'二元词典构建失败{e}')
            return False

        return True
    
    @staticmethod
    def reload():
        path = NLPConfig.BiGramDictionaryPath
        return CoreBiGramTableDictionary.load(path)
    
    @staticmethod
    def getBiFrequency(a, b):
        """
        获取共现频次
        -param a 第一个词
        -param b 第二个词
        return 第一个词@第二个词出现的频次
        """
        self = CoreBiGramTableDictionary

        if isinstance(a, int):
            idA = -1 if a < 0 else a
        else:
            idA = self.coreDictionary.trie.exactMatchSearch(a)

        if idA < 0:
            return 0
       

        if isinstance(b, int):
            idB = -1 if b < 0 else b
        else:
            idB = self.coreDictionary.trie.exactMatchSearch(b)

        if idB < 0:
            return 0
        
        # 在 [self.start[idA], self.start[idA + 1] - self.start[idA]] 区间内使用二分法查找第二单词编号
        # 之所以需要使用区间查询，是因为在pair中存放的第二单词会是重复的，比如“商品@和”以及“服务@和”第一单词不同但是第二单词相同。
        # self.start[idA] 实际上保存的是上一个单词的offset值
        index = self.binarySearch(self.pair, self.start[idA], self.start[idA + 1] - self.start[idA], idB)
        if index < 0:
            return 0
        
        return self.pair[index * 2 + 1]
    
    @staticmethod
    def binarySearch(a, fromIndex, length, key):
        """
        二分搜索，由于二元接续前一个词固定时，后一个词比较少，所以二分也能取得很高的性能
        -param a 目标数组
        -param fromIndex 开始下标, 单词在双数组内base的值
        -param length 长度
        -param key 词的id
        return 共现频次
        """
        low  = fromIndex
        high = fromIndex + length - 1

        while low <= high:
            mid = (low + high) // 2
            midVal = a[mid * 2]

            if midVal < key:
                low = mid + 1
            elif midVal > key:
                high = mid - 1
            else:
                return mid

        return -(low + 1)
