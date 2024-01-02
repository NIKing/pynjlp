#import sys
#import os
#sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import getFileList
from nlp.corpus.document.Document import Document
from nlp.corpus.document.sentence.Sentence import Sentence

class CorpusLoader():
    def __init__(self):
        pass

    @staticmethod
    def convert2DocumentList(folderPath):
        documentList = []
        fileList = getFileList(folderPath)

        for file in fileList:
            document = CorpusLoader.convert2Document(file)
            documentList.append(document)

        return documentList
    
    @staticmethod
    def convert2SentenceList(path):
        simpleList = []
        documentList = CorpusLoader.convert2DocumentList(path)
        
        for document in documentList:
            for sentence in document.sentenceList:
                simpleList.append(sentence.wordList)

        return simpleList
    
    @staticmethod
    def convert2Document(file):
        document = Document.create(file)
        if not document:
            print(f'【{file}】读取失败')

        return document

