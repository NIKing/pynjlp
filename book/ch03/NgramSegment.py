import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import readlinesText

def train_bigram(corpus_path, model_path):
    """训练n元语法""" 
    sents = readlinesText(corpus_path) 
    new_sentens = []
    for sent in sents:

        words = sent.split(' ')
        new_words = []
        for word in words:
            if word.find('/') == -1:
                word += '/n'
                new_words.append(word)

        new_sentens.append(new_words)

    print(new_sentens)

    maker = NatureDictionaryMaker()
    maker.compute(new_sentens)
    maker.saveTo(model_path)

if __name__ == '__main__':
    msr_train = '/pynjlp/data/my_cws_corpus.txt'
    msr_model = '/pynjlp/data/test/my_cws_model'

    train_bigram(msr_train, msr_model)
    #segment = load_bigram(msr_model)

