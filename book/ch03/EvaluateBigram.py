import sys
sys.path.append('/pynjlp')

from book.ch03.NgramSegment import train_bigram, load_bigram
from src.corpus.MSR import MSR
from nlp.seg.common.CWSEvaluator import CWSEvaluator 

if __name__ == '__main__':
    MSR_MODEL_PATH = MSR.MODEL_PATH + '_ngram'

    #train_bigram(MSR.TRAIN_PATH, MSR_MODEL_PATH)
    segment = load_bigram(MSR_MODEL_PATH)
    
    result = CWSEvaluator.evaluate(segment, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)
    print(result)
