import sys
sys.path.append('/pynjlp')

from src.corpus.MSR import MSR

from nlp.seg.other.DoubleArrayTrieSegment import DoubleArrayTrieSegment
from nlp.seg.common.CWSEvaluator import CWSEvaluator 

from nlp.corpus.io.IOUtil import loadDictionary

if __name__ == '__main__':
    dictionary = loadDictionary(MSR.TRAIN_WORDS)
    segment = DoubleArrayTrieSegment(dictionary)

    result = CWSEvaluator.evaluate(segment, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)
    print(list(zip(['P', 'R', 'F1', 'OOV-R', 'IV-R'], list(result))))
