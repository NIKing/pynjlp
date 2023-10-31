import sys
sys.path.append('pynjlp')

from nlp.collection.trie.bintrie.BinTrie import BinTrie
from nlp.corpus.io.IOUtil import writeTxtByList 
from nlp.corpus.dictionary.item.Item import Item

"""一个通用的词典制作工具，词条格式：词 标签 词频"""
class DictionaryMaker():

    def __init__(self):
        # 以字典树收集词典
        self.trie = BinTrie()
    
    def addWord(self, word):
        item = self.trie.get(word.getValue())
        if not item:
            item = Item(word.getValue(), word.getLabel())
            self.trie.put(item.key, item)
        else:
            # 实际上这是增加相同标签的频次
            item.addLabel(word.getLabel()) 
        
    def saveTxtTo(self, path):
        if self.trie.getSize() <= 0:
            return True

        try:
            # 获取词典条目(在addWord中put到字典树里了)，并保存
            entries = self.trie.entrySet()
            entries = [entry for entry in entries.values()]
            
            print(path, entries)
            writeTxtByList(path, entries)
        except Exception as e:
            print(f'保存到【{path}】失败【{e}】')
            return False

        return True

            
        
