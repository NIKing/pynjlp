import sys
sys.path.append('/pynjlp')

from nlp.corpus.document.CorpusLoader import CorpusLoader
from nlp.corpus.dictionary.NatureDictionaryMaker import NatureDictionaryMaker

def train_bigram(corpus_path, model_path):
    """训练n元语法""" 
    sents = CorpusLoader.convert2SentenceList(corpus_path)

    for sent in sents:
        for word in sent:
            
            if not word.label:
                word.setLabel('n')
    
    
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    maker.saveTxtTo(model_path)

if __name__ == '__main__':
    msr_train = '/pynjlp/data/my_cws_corpus'
    msr_model = '/pynjlp/data/test/my_cws_model'

    train_bigram(msr_train, msr_model)
    #segment = load_bigram(msr_model)

