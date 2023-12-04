
import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import loadDictionary

from nlp.collection.trie.DoubleArrayTrieSearcher import Searcher
from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie
from nlp.collection.trie.bintrie.HashCode import hash_code

def createTinyDictionary():
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
    #print(dat.get('自然语言'))
    print(dat.get('上海市'))
    #print(dat.get('三三两两'))

def test_2():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
    dat = DoubleArrayTrie(dictionary)

    #text = '江西鄱阳湖干枯，中国最大淡水湖变成大草原'
    #res = dat.parseText(text)

    text = '我变成大魔王'
    text = '上海市虹口区大连西路550号SISU'
    res = dat.parseLongestText(text)
    #res  = dat.matchLongest(text)

    print(res)

def test_3():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.4.txt')
    dat = DoubleArrayTrie(dictionary)
    
    print(dat.toString())
    #print(dat.get('上海市'))

if __name__ == '__main__':

    arguments = sys.argv
    debug = arguments[1] if len(arguments) > 1 else '1'
    
    if debug == '-1':
        test_1()
    elif debug == '-2':
        test_2()
    elif debug == '-3':
        test_3()


    



