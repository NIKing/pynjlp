import re
#import sys
#sys.path.append('/pynjlp')

from nlp.corpus.document.sentence.word.Word import Word
from nlp.corpus.document.sentence.word.WordFactory import WordFactory
from nlp.corpus.document.sentence.word.CompoundWord import CompoundWord

class Sentence:
    def __init__(self, wordList = []):
        self.wordList = wordList

        self.offset = 0
    
    def __len__(self):
        return len(self.wordList)

    def __iter__(self):
        for word in self.wordList:
            yield word

    @staticmethod
    def create(param):
        if not param:
            return None
        
        # 下面的正则表达式比较长，但是，核心正则是 [^\s\]]+[0-9a-zA-Z]+ 表示匹配以空格和]开头的数字和字母（必须是两个字符以上才可以匹配）
        # 整个表达式表示，可以匹配多个这个情况
        #pattern = re.compile("(\\[(([^\\s\\]]+/[0-9a-zA-Z]+)\\s+)+?([^\\s\\]]+/[0-9a-zA-Z]+)]/?[0-9a-zA-Z]+)|([^\\s]+/[0-9a-zA-Z]+)")

        param = param.strip()
        pattern = re.compile("((([^\s\]]+[0-9a-zA-Z]+)\s+)+?(([^\s\]]+[0-9a-zA-Z]+)\s+)?[0-9a-zA-Z]+)|[^\s\]]+[0-9a-zA-Z]+")
        matcher = pattern.match(param)
        #print(f'--{param}--{matcher}')

        wordList = []
        if matcher:
            for single in matcher.group():
                single = matcher.group()
                word = WordFactory.create(single)

                if not word:
                    print(f'在用【{single}】构造句子失败')
                    return None

                wordList.append(word)
        
        # 按照无词性来解析，通过正则分割句子（\s 空格分割符）
        if len(wordList) <= 0:
            for w in re.split(r'\s+', param):
                wordList.append(Word(w))

        return Sentence(wordList)


    def toSimpleWordList(self) -> list:
        wordList = []

        for word in self.wordList:
            if isinstance(word, CompoundWord):
                wordList.extend(word.innerList)
            else:
                wordList.append(word)

        return wordList


