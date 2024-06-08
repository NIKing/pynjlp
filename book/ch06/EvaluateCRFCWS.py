import sys
sys.path.append('/pynjlp')

from src.corpus.MSR import MSR

from nlp.model.crf.CRFSegmenter import CRFSegmenter
#from nlp.model.crf.CRFLexicalAnalyzer import CRFLexicalAnalyzer

if __name__ == "__main__":

    segmenter = CRFSegmenter()
    segmenter.train(MSR.TRAIN_PATH, "/pynjlp/data/test/crf-cws-model")
    

