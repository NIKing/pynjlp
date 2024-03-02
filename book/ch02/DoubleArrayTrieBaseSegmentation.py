
import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import loadDictionary

from nlp.collection.trie.DoubleArrayTrieSearcher import Searcher
from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie
from nlp.collection.trie.bintrie.HashCode import hash_code

from nlp.seg.other.DoubleArrayTrieSegment import DoubleArrayTrieSegment

def createTinyDictionary():
    return {
        "自然": "nature",
        "自然人": "human",
        "自然语言": "language",
        "自语": "talk to oneself",
        "入门": "introduction",
    }

    return {
        "±": "w",
        "三七开": "san qi kai",
        "三三两两": "shan shan liang liang",
        "上海": "shang hai",
        "上海市": "shang hai shi",
    }

    return {
        "入口": "entry",
        "自然": "nature",
        "自然人": "human",
        "自然语言": "language",
        "自语": "talk to oneself",
        "入门": "introduction",
        "魔王": 'devil',
        "变成": "change",
    }
    


def test_1():
    dictionary = createTinyDictionary()
    dat = DoubleArrayTrie(dictionary)
    
    print(dat.toString())
    #print(dat.parseText('自然语言处理'))
    #print(dat.parseLongestText('自然语言处理'))
    #print(dat.get('上海市'))
    #print(dat.get('三三两两'))

def test_2():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
    dat = DoubleArrayTrie(dictionary)

    text = '江西鄱阳湖干枯，中国最大淡水湖变成大草原'
    #res = dat.parseText(text)

    #text = '我变成大魔王'
    #text = '上海市虹口区大连西路550号SISU'
    res = dat.parseLongestText(text)
    #res  = dat.matchLongest(text)

    print(res)

def test_3():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.4.txt')
    dat = DoubleArrayTrie(dictionary)
    
    print(dat.toString())
    #print(dat.get('上海市'))

def test_4():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
    segment = DoubleArrayTrieSegment(dictionary)

    text = '江西鄱阳湖干枯，中国最大淡水湖变成大草原'
    res = segment.seg(text)
    print(res)

def test_5():
    dictionary = loadDictionary('/pynjlp/data/test/my_cws_model.txt')
    segment = DoubleArrayTrieSegment(dictionary)

    text = '商品和服务'
    res = segment.seg(text)
    print(res)

def test_6():
    dictionary = loadDictionary('/pynjlp/data/test/my_cws_model.txt')
    dat = DoubleArrayTrie(dictionary)

    text = '商品和服务'
    res = dat.parseText(text)
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
    elif debug == '-5':
        test_5()
    elif debug == '-6':
        test_6()


    



