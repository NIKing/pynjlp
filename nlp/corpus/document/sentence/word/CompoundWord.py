from nlp.corpus.document.sentence.word.Word import Word

"""复合词，由两个或以上的词组成"""
class CompoundWord():
    def __init__(self, innerList = [], label = ""):
        self.innerList = innerList
        self.label = label
    
    @staticmethod
    def create(param):
        if not param:
            return None

        cutIndex = param.rfind(']')
        if cutIndex == -1 or curIndex <= 2 or cutIndex == len(param) - 1:
            return None

        wordList = []
        wordParam = param[:cutIndex]
        
        # 获取复合词中每个词
        for single in wordParam.split('\\s+'):
            if len(single) == 0:
                continue

            word = Word.create(single)
            if not word:
                print(f'使用参数{single}构建Word失败')
                return

            wordList.append(word)

        # 获取词性/标签
        labelParam = param[(cutIndex + 1):]
        if labelParam.startswith('/'):
            labelParam = labelParam[1:]

        return CompoundWord(wordList, labelParam)

