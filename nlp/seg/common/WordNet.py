from nlp.seg.common.Vertex import Vertex
from nlp.seg.common.Graph import Graph

from nlp.corpus.tag.Nature import Nature

from nlp.dictionary.other.CharType import CharType
from nlp.dictionary.CoreDictionary import CoreDictionary, Attribute

from nlp.utility.Predefine import Predefine
from nlp.utility.MathUtility import MathUtility

class WordNet():
    
    # 节点，每一行都是前缀词，跟图的表示方式不同
    vertexes = []
    
    # 共有多少个节点
    size = 0
    
    # 原始句子
    sentence = ""
    
    # 原始句子对应的数组
    charArray = []

    def __init__(self, sentence):
        self.charArray = list(sentence)
        
        # 需要给vertexes固定长度
        self.vertexes = [[] for i in range(len(self.charArray) + 2)]

        # 添加开始和结尾标记
        self.vertexes[0].append(Vertex.newB())
        self.vertexes[len(self.vertexes) - 1].append(Vertex.newE())
        #print(self.vertexes)

        self.size = 2

    def add(self, line, vertex):
        """
        添加顶点
        -param line 行号
        -param vertex 顶点
        """
        for oldVertex in self.vertexes[line]:
            # 保证唯一性
            if len(oldVertex.realWord) == len(vertex.realWord):
                return
        
        self.vertexes[line].append(vertex)
        self.size += 1

    def addAtom(self, line, atomSegment):
        """
        添加顶点，由原子分词顶点添加
        -param line 所在行号
        -param atomSegment
        """
        offset = 0
        for atomNode in atomSegment:
            sWord  = atomNode.sWord
            nature = Nature.n
            nPOS   = atomNode.nPOS

            id = -1
            
            if nPOS == CharType.CT_CHINESE:
                break

            if nPOS == CharType.CT_NUM or nPOS == CharType.CT_INDEX or nPOS == CharType.CT_CNUM:
                nature = Nature.m
                sWord  = Predefine.TAG_NUMBER
                id     = CoreDictionary.M_WORD_ID
                break

            if nPOS == CharType.CT_DELIMITER or nPOS == CharType.CT_OTHER:
                nature = Nature.w
                break

            if nPOS == CharType.CT_SINGLE:
                nature = Nature.nx
                sWord  = Predefine.TAG_CLUSTER
                id     = CoreDictionary.X_WORD_ID
                break
            
            #print(sWord, atomNode.sWord, atomNode)
            self.add(line + offset, Vertex(sWord, atomNode.sWord, Attribute(nature, Predefine.OOV_DEFAULT_FREQUENCY), id))

            offset += len(atomNode.sWord)

    
    def size(self):
        return self.size

    def getVertexes(self):
        """获取内部顶点表格"""
        return self.vertexes

    def getVertexesLineFirst(self):
        """获取顶点数组(实际上，就是展开二维数组)，按行优先列次之的顺序构造的顶点数组返回"""
        
        vertexes = []
        i = 0
        for vertexList in self.vertexes:
            for vertex in vertexList:
                vertex.index = i
                vertexes.append(vertex)
                i += 1

        return vertexes

        
    def toString(self):
        line, sb = 0, []
        for vertexList in self.vertexes:
            sb.append(f'{line} : {[vertex.toString() for vertex in vertexList]}')
            sb.append("\n")
            line += 1

        return ''.join(sb)


    def toGraph(self):
        """词网转词图"""
        graph = Graph(self.getVertexesLineFirst())

        for row in range(len(self.vertexes) - 1):
            vertexListFrom = self.vertexes[row]

            for _from in vertexListFrom:
                assert len(_from.realWord) > 0, "空节点会导致死循环"
                toIndex = row + len(_from.realWord)

                for to in self.vertexes[toIndex]:
                    graph.connect(_from.index, to.index, MathUtility.calculateWeight(_from, to))

        return graph
