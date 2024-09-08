import sys
sys.path.append('/pynjlp')

from src.corpus.MSR import MSR

from nlp.model.crf.CRFSegmenter import CRFSegmenter
#from nlp.model.crf.CRFLexicalAnalyzer import CRFLexicalAnalyzer

if __name__ == "__main__":
    
    segmenter = CRFSegmenter(None)  # 传入None代表是训练模式
    segmenter.train(MSR.TRAIN_PATH, "/pynjlp/data/test/crf-cws-model")
    exit()

    #segmenter = CRFSegmenter('/pynjlp/data/test/crf-cws-model.txt')
    #res = segmenter.segment('商品和服务')
    #print(res)
    

