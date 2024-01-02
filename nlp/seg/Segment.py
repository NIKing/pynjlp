from abc import ABC, abstractmethod

from nlp.seg.Config import Config
from nlp.seg.common.Term import Term
from nlp.seg.nShort.path.AtomNode import AtomNode

from nlp.dictionary.CustomDictionary import CustomDictionary

class Segment(ABC):
    
    config = None

    customDictionary = CustomDictionary.DEFAULT

    def __init__(self):
        self.config = Config()
    
    @abstractmethod
    def segSentence(self, sentence = str) -> list:
        pass
    
    def seg(self, sentence = str) -> list:
        return [term.toString() for term in self.segSentence(sentence)]

    def enableCustomDictionary(self, customDictionary):
        """是否启用用户字典"""
        self.config.useCustomDictionary = True
        self.customDictionary = customDictionary
        
        return self

    def enableAllNameEntityRecognize(self, enable):
        """否启用所有的命名实体识别"""
        self.config.nameRecognize = enable
        self.config.japaneseNameRecognize = enable
        self.config.translatedNameRecognize = enable
        self.config.placeRecognize = enable
        self.config.organizationRecognize = enable

        self.config.updateNerConfig();
        return self; 
    
    @staticmethod
    def quickAtomSegment(charArray, start, end):
        """快速原子分词, 如果字符串中前后字符类型一致则合并，否则分开"""
        offsetAtom = start
        preType = type(charArray[offsetAtom])
        curType = ""

        atomNodeList = []
        while offsetAtom < end:
            curType = type(charArray[offsetAtom])

            if curType != preType:

                # 浮点数识别，上一个是浮点数，下一个也是浮点数，当前字符就不拆分
                if isinstance(preType, 'float') and '，,．.'.indexOf(charArray[offsetAtom]) != -1:
                    if offsetAtom + 1 < end:
                        nextType = type(charArray[offsetAtom + 1])
                        if isinstance(nextType, 'float'):
                            continue
                
                atomNodeList.append(AtomNode(''.join(charArray[start:offsetAtom]), preType))
                start = offsetAtom

            preType = curType
            offsetAtom += 1

        if offsetAtom == end:
            atomNodeList.append(AtomNode(''.join(charArray[start:offsetAtom]), preType))

        return atomNodeList

    def convert(self, vertexList, offsetEnabled = False):
        """
        将一条路径转为最终结果
        -param vertexList 节点列表
        -param offsetEnabled 是否计算offset
        """
        assert vertexList != None
        assert vertexList.size() >= 2, "这条路径不应当短于2" + vertexList.toString()

        length = vertexList.size() - 2
        iterator = vertexList.iterator()
        iterator.next()
        
        resultList = []
        if offsetEnabled:
            offset = 0
            for i in range(length):
                vertex = iterator.next()
                term = segment.vertexConvertTerm(vertex)
                term.offset = offset

                offset += term.length()
                resultList.append(term)

        else:
            for i in range(length):
                vertex = iterator.next()
                term = Segment.vertexConvertTerm(vertex)

                resultList.append(term)

        return resultList

    
    @staticmethod
    def vertexConertTerm(self, vertex):
        """将节点转换term"""
        return Term(vertex.realWord, vertex.guessNature())
