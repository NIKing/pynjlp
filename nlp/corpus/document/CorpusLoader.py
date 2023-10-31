import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import readlinesTxt
from nlp.corpus.document.Document import Document

class CorpusLoader():
    def __init__(self):
        pass
    
    @staticmethod
    def convert2DocumentList(file):
        document_list = []
        file_data_list = readlinesTxt(file)
        
        for file_data in file_data_list:
            document_list.append(Document.create(file_data))

        return document_list
    
    @staticmethod
    def convert2SentenceList(path):
        simple_list = []
        document_list = CorpusLoader.convert2DocumentList(path)

        for document in document_list[:1]:
            for sentence in document.sentenceList:
                simple_list.append(sentence.wordList)

        return simple_list
