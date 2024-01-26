import sys
sys.path.append('/pynjlp')

from nlp.corpus.document.CorpusLoader import CorpusLoader
from nlp.corpus.dictionary.NatureDictionaryMaker import NatureDictionaryMaker

# 记录下，import 也会造成 class 在定义类属性时候执行实例化对象加载
from nlp.dictionary.CoreDictionary import CoreDictionary
from nlp.dictionary.CoreBiGramTableDictionary import CoreBiGramTableDictionary

from nlp.seg.dijkstra.DijkstraSegment import DijkstraSegment
from nlp.seg.viterbi.ViterbiSegment import ViterbiSegment

from nlp.NLP import NLPConfig

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

def load_bigram(model_path, viterbi = False):
    """加载模型"""
    NLPConfig.CoreDictionaryPath = model_path + '.txt'
    NLPConfig.BiGramDictionaryPath = model_path + '.ngram.txt'

    # 一元模型加载
    CoreDictionary.reload()

    # n元模型加载
    CoreBiGramTableDictionary.reload()
    
    return  ViterbiSegment() if viterbi else DijkstraSegment()


msr_train = '/pynjlp/data/my_cws_corpus'
msr_model = '/pynjlp/data/test/my_cws_model'


def test_1():
    train_bigram(msr_train, msr_model)

def test_2():
    NLPConfig.CoreDictionaryPath = msr_model + '.txt'
    NLPConfig.BiGramDictionaryPath = msr_model + '.ngram.txt'

    # 一元模型加载
    CoreDictionary.reload()

    # n元模型加载
    CoreBiGramTableDictionary.reload()

    #print(f'【商品】的频次：{CoreDictionary.getTermFrequency()}')
    #print(f"【商品@和】的频次：{CoreBiGramTableDictionary.getBiFrequency('商品', '和')}")
    
    # 最短路径分词
    segment = DijkstraSegment().enableAllNameEntityRecognize(False)
    #res = segment.seg('商品和服务')
    res = segment.seg('货币和服务')

    print(res)

def test_3():
    NLPConfig.CoreDictionaryPath = msr_model + '.txt'
    NLPConfig.BiGramDictionaryPath = msr_model + '.ngram.txt'
   
    # 一元模型加载
    CoreDictionary.reload()

    # n元模型加载
    CoreBiGramTableDictionary.reload()

    coreBiGramTableDictionary = CoreBiGramTableDictionary()
    
    #res = coreBiGramTableDictionary.getBiFrequency('始##始', '商品')
    res = coreBiGramTableDictionary.getBiFrequency('和服', '物美价廉')
    print(res)

def test_4():
    NLPConfig.CoreDictionaryPath = msr_model + '.txt'
    NLPConfig.BiGramDictionaryPath = msr_model + '.ngram.txt'
    
    # 一元模型加载
    CoreDictionary.reload()
    
    # n元模型加载
    CoreBiGramTableDictionary.reload()

    #print(f'【商品】的频次：{coreDictionary.getTermFrequency()}')
    #print(f"【商品@和】的频次：{CoreBiGramTableDictionary.getBiFrequency('商品', '和')}")
    
    # 最短路径分词
    segment = ViterbiSegment().enableAllNameEntityRecognize(False)
    res = segment.seg('货币和服务')

    print(res)


if __name__ == '__main__':
    arguments = sys.argv
    debug = arguments[1] if len(arguments) > 1 else '1'
    
    if debug == '-1':
        test_1()
    elif debug == '-2':
        test_2()
    elif debug == '-3':
        test_3()
    elif debug == '-4':
        test_4()

