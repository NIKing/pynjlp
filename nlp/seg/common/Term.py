"""
一个单词，用户可以直接访问此单词的全部属性
"""
class Term():
    
    # 词语
    word = ''
    
    # 词性
    nature = ''

    # 在文本中的起始位置
    offset = 0

    def __init__(self, word, nature):
        self.word = word
        self.nature = nature

    def toString(self):
        return self.word
