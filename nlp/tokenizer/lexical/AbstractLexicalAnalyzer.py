from nlp.seg.CharacterBasedSegment import CharacterBasedSegment
from nlp.seg.common.Term import Term

from nlp.dictionary.other.CharTable import CharTable
from nlp.dictionary.CoreDictionary import CoreDictionary

from nlp.corpus.tag.Nature import Nature

"""结构化词法分析器"""
class AbstractLexicalAnalyzer(CharacterBasedSegment):
    def __init__(self, segmenter = None, posTagger = None, neRecognizer = None):
        super().__init__()

        self.segmenter = segmenter
        self.posTagger = posTagger
        self.neRecognizer = neRecognizer

        self.enableRuleBasedSegment = False
        self.charTable = []

        if posTagger:
            self.config.speechTagging = True

        if neRecognizer:
            self.config.ner = True

    def segSentence(self, sentence = str):
        if len(sentence) <= 0:
            return []
        
        normalized = CharTable.normalization(sentence)
        
        wordList = []
        attributeList = self.segmentWithAttribute(sentence, normalized, wordList)

        termList = []
        for i, word in enumerate(wordList):
            term = Term(word, None)
            term.offet = i

            termList.append(term)

        if not self.config.speechTagging:
            return termList

        if not self.posTagger:
            for term in termList:
                attribute = CoreDictionary.get(term.word)
                
                if not attribute:
                    term.nature = attribute.nature[0]
                else:
                    term.nature = Nature.n

        return termList 

    def roughSegSentence(self, sentence):
        return None
    
    def segmentAfterRule(self, sentence, normalized, wordList):
        """基于规则的分词""" 
        if not self.enableRuleBasedSegment:
            return self.segmenter.segment(sentence, normalized = normalized, output = wordList)
        
        start, end = 0, 0
        curType, preType = '', self.typeTable[normalized[end]]
        
        while end < len(normalized):
            curType = self.typeTable[normalized[end]]

            if preType != curType:

                # 数字类型
                if preType == CharType.CT_NUM:

                    # 浮点数
                    if "，,．.".find(normalized[end]) != -1:
                        
                        if end + 1 >= len(normalized):
                            break

                        if self.typeTable[normalized[(end + 1)]] == CharType.CT_NUM:
                            continue

                    elif "年月日时分秒".find(normalized[end]) != -1:
                        preType = curType
                        continue
                
                self.pushPiece(sentence, normalized, start, end, preType, wordList)
                start = end

            preType = curType
            end += 1

        if end == len(normalized):
            self.pushPiece(sentence, normalized, start, end, preType, wordList)


    def segmentWithAttribute(self, original, normalized, wordList):
        """返回用户词典中的attribute的分词"""
        
        attributeList = []
        if self.config.useCustomDictionary:
            if self.config.forceCustomDictionary:
                self.segment(original, normalized, attributeList)
            else:
                self.segmentAfterRule(original, normalized, wordList)
                attributeList = self.combineWithCustomDictionary(wordList)

        else:
            print('----')
            self.segmentAfterRule(original, normalized, wordList)
            attributeList = None

        return attributeList
    

    def setEnableRuleBasedSegment(enableRuleBasedSegment):
        """是否执行规则分词（英文数字标点等的规则预处理）。规则永远是丑陋的，默认关闭。"""
        self.enableRuleBasedSegment = enableRuleBasedSegment
