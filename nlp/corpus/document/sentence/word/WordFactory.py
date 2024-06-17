from nlp.corpus.document.sentence.word.Word import Word
from nlp.corpus.document.sentence.word.CompoundWord import CompoundWord

"""一个单词工厂类，可以生成不同类型词语"""
class WordFactory():
    
    @staticmethod
    def create(param):
        if not param:
            return None

        if param.starswith('[') and not param.startwith('[/'):
            return CompoundWord.create(param)

        return Word.create(param)
