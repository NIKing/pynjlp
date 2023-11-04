import sys
sys.path.append('/pynjlp')

from nlp.corpus.dictionary.CommonDictionaryMaker import CommonDictionaryMaker

from nlp.utility.Perdefine import Perdefine 
from nlp.corpus.tag.Nature import Nature

from nlp.corpus.document.sentence.word.Word import Word

class NatureDictionaryMaker(CommonDictionaryMaker):
    def __init__(self):
        super().__init__()
        self.sentenceList = []

    def roleTag(self):
        """这里的角色标注，主要是标注每个句子的【开始】和【结束】"""
        print('开始角色标注')
        
        new_sentence_list = []
        for i, sentence in enumerate(self.sentenceList):
            sentence.insert(0, Word(Perdefine.TAG_BEGIN, Nature.begin))
            sentence.append(Word(Perdefine.TAG_END, Nature.end))

            new_sentence_list.append(sentence)
        
        self.sentenceList = new_sentence_list


    def addDictionary(self):
        print('开始制作词典')

        for wordList in self.sentenceList:
            # 上一个单词
            pre = None

            for word in wordList:
                self.dictionaryMaker.addWord(word)
                
                # 二元语法词典
                if pre:
                    self.nGramDictionaryMaker.addPair(pre, word)

                pre = word

