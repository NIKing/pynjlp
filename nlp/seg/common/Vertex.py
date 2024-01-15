from nlp.dictionary.CoreDictionary import CoreDictionary, Attribute
from nlp.utility.Predefine import Predefine
from nlp.corpus.tag.Nature import Nature

class Vertex():

    # 节点对应的词或等效词（如未##数）
    word = " "
    
    # 节点对应的真实词，绝对不含##
    realWord = " "
    
    # 词的属性，谨慎修改属性内部的数据，因为会影响到字典<br>
    attribute = None
    
    # 等效词ID,也是Attribute的下标
    wordID = -1
    
    # 在一维顶点数组中的下标，可以视作这个顶点的id
    index = -1
    
    # 到该节点的最短路径的前驱节点
    _from = None
    
    # 最短路径对应的权重
    weight = 0

    def __init__(self, word, realWord, attribute, wordID = -1):
        if not attribute:
            attribute = Attribute(Nature.n, 1)

        self.wordID = wordID
        self.attribute = attribute
        
        #print(f'word={word}, realword={realWord}, {attribute}, {wordID}')
        if not word:
            word = self.compileRealWord(realWord, attribute)

        assert len(realWord) > 0, "构造空白节点会导致死循环！"

        self.word = word
        self.realWord = realWord

    def compileRealWord(self, realWord, attribute):
        pass

    def getAttribute(self):
        """获取词的属性"""
        return self.attribute
    
    @staticmethod
    def newPunctuationInstance(realWord):
        """创建一个标点符号实例"""
        return Vertex(Predefine.TAG_PLACE, realWord, Attribute(Nature.ns, 1000))

    @staticmethod
    def newB():
        """生成线程安全的起始节点"""
        wordId = CoreDictionary.getWordID(Predefine.TAG_BEGIN)
        return Vertex(Predefine.TAG_BEGIN, " ", Attribute(Nature.begin, Predefine.TOTAL_FREQUENCY / 10), wordId)
    
    @staticmethod
    def newE():
        """生成线程安全的终止节点"""
        wordId = CoreDictionary.getWordID(Predefine.TAG_END)
        return Vertex(Predefine.TAG_END, " ", Attribute(Nature.end, Predefine.TOTAL_FREQUENCY / 10), wordId)

    
    def guessNature(self):
        return self.attribute.nature[0]

    def toString(self):
        return self.realWord if self.realWord != ' ' else self.word
    

