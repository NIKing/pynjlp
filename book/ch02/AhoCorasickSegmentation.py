import sys
sys.path.append('/pynjlp')

from nlp.corpus.io.IOUtil import loadDictionary
from nlp.algorithm.ahocorasick.trie.Trie import Trie

from nlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie import AhoCorasickDoubleArrayTrie
from nlp.seg.other.AhoCorasickDoubleArrayTrieSegment import AhoCorasickDoubleArrayTrieSegment

def test_1():
    keywords = ["hers", "his", "she", "he"]
    #keywords = sorted(keywords)

    print(keywords)
    trie = Trie(keywords)

    for emit in trie.parseText('ushers'):
        print(emit[0], emit[1], emit[2])
    
def test_2():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
    dictionary_keys = list(dictionary.keys())
    
    trie = Trie(dictionary_keys)
    txt = '江西鄱阳湖干枯，中国最大淡水湖变成大草原'
    
    wordList = []
    for emit in trie.parseText(txt):
        wordList.append(emit[2])

    print(wordList)

def test_3():
    keywords = ["hers", "his", "she", "he"]
    treeMap = {}
    for keyword in keywords:
        treeMap[keyword] = keyword

    res = AhoCorasickDoubleArrayTrie(treeMap).parseText('ushers')
    print(res)

def test_4():
    dictionaryPaths = ['/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt']
    acdats = AhoCorasickDoubleArrayTrieSegment(dictionaryPaths)

    res = acdats.seg('江西鄱阳湖干枯')
    print(res)


def test_5():
    text = "今天，天气很好，我自己正在学习自然语言处理技术"
    dicts = ['自然', '自然语言处理', '自己', '我自己'] # , '自己正'
    
    print(dicts)

    trie = Trie(dicts)

    words = []
    for emit in trie.parseText(text):
        words.append(emit)

    print(words)

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


