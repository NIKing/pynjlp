import re
import sys
sys.path.append('/pynjlp')

from nlp.corpus.document.sentence.Sentence import Sentence

class Document:
    def __init__(self, sentenceList):
        self.sentenceList = sentenceList
    
    @staticmethod
    def create(param):

        # 表达式是从java的代码中复制的，但是好像有些问题，/w 应该是要 \w 的写法
        # 整个表达式的意思，想要匹配句子中(。！？\n $)后面跟着的字母，其实就是找词性
        #pattern = re.compile(".+?((。/w)|(！/w )|(？/w )|\\n|$)")
        pattern = re.compile(".+?((。\w)|(！\w)|(？\w)|\n|$)")
        #matcher = pattern.findall(param)

        matcher = pattern.match(param)
        #print(f'===={matcher.group()}')

        sentenceList = []
        for single in matcher.group():
            sentence = Sentence.create(single)

            if not sentence:
                print(f"使用 【{single}】 构建句子失败")
                return

            sentenceList.append(sentence)
        
        
        if len(sentenceList) <= 0:
            sentenceList = [Sentence()]
        
        #print(f'sentence==={sentenceList}==={len(sentenceList)}==')
        return Document(sentenceList)


