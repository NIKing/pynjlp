import sys
sys.path.append('/pynjlp')

from pyhanlp import *
from nlp.corpus.io.IOUtil import loadDictionary
from nlp.collection.trie.bintrie.BinTrie import BinTrie

IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')
BinTrie2 = JClass('com.hankcs.hanlp.collection.trie.bintrie.BinTrie')


def parseLongestText(trie, text):
    """匹配长文本，根据字典返回最长匹配，比如'工'和'工信部'，返回'工信部'""" 
    i, text_len, word_list = 0, len(text), []

    while i < text_len:
        state = trie.transition(text[i])
        if state:
            to  = i + 1
            end = to
            value = state.getValue()

            for j, char in enumerate(text[to:]):
                state = state.transition(char)
                if state == None:
                    break
                
                #print(f'--i={i},,j={j}, value={state.getChar()}')
                #print(state.getValue())
                value = state.getValue()
                if value:
                    end = to + j + 1

            if value:
                word_list.append(text[i:end])
                i += end - 1
                continue
        
        i += 1

    return word_list

def test_1():
    dict_path = '/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt'
    dict1 = IOUtil.loadDictionary([dict_path])
    trie = BinTrie2(dict1)

    text = '江西潘阳湖干枯，中国最大淡水湖变成大草原'
    print(parseLongestText(trie, text))


def test_2():
    dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
    #dictionary = loadDictionary('/pynjlp/data/dictionary/test_dict.txt')
    binTrie = BinTrie(dictionary)

    #text = '工信处发布说明'
    text = '江西潘阳湖干枯，中国最大淡水湖变成大草原'

    #word_list_1 = binTrie.parseText(text)
    #print(word_list_1)

    word_list_2 = binTrie.parseLongestText(text)
    print('结果', word_list_2)

if __name__ == '__main__':

    arguments = sys.argv
    debug = arguments[1] if len(arguments) > 1 else '1'
    
    if debug == '-1':
        test_1()
    elif debug == '-2':
        test_2()


