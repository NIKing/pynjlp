import sys
sys.path.append('/pynjlp')

from .CommonDictionaryMaker import CommonDictionaryMaker

from nlp.utility.Perdefine import Perdefine 
from nlp.tag.Nature import Nature

class NatrueDictionaryMaker(CommonDictionaryMaker):
    def __init__(self):
        super().__init__(self)

    def roleTag(self, sentence_list):
        print('开始角色标注')
        
        new_sentence_list = []
        for i, sentence in enumerate(sentence_list):
            print(f'{i}/{len(sentence_list)}')

             new_sentence_list.append(sentence)
             new_sentence_list.append([Perdefine.TAG_BEGIN, Natue.begin])
             new_sentence_list.append([Perdefind.TAG_END, Natue.end])
        
        return new_sentence_list


    def addDictionary(self, sentence_list):
        print('开始制作词典')

        for word_list in sentence_list:
            
            # 前缀
            pre = None
            for word in word_list:
                self.dictionaryMaker.add(word)

                if pre:
                    self.nGramDictionaryMaker.addPair(pre, word)

                pre = word

