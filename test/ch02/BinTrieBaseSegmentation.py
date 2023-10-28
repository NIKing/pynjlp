import sys
sys.path.append('/pynjlp')

from nlp.collection.corpus.io.IOUtil import loadDictionary
from nlp.collection.trie.bintrie.BinTrie import BinTrie

#dictionary = loadDictionary('/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt')
dictionary = loadDictionary('/pynjlp/data/dictionary/test_dict.txt')
binTrie = BinTrie(dictionary)

text = '江西潘阳湖干枯，中国最大淡水湖变成大草原'
text = '工信处发布说明'

#word_list_1 = binTrie.parseText(text)
#print(word_list_1)

word_list_2 = binTrie.parseLongText(text)
print('结果', word_list_2)


