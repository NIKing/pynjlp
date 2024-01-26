import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import readlinesText
# 只是读取文件，暂时用不到convert2SentenceList

if __name__ == '__main__':
    corpus_path = '/pynjlp/data/my_cws_corpus.txt'
    sents = readlinesText(corpus_path)
    for sen in sents:
        print(sen)

