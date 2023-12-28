from nlp.collection.trie.DoubleArrayTrie import DoubleArrayTrie
from nlp.collection.trie.bintrie.BinTrie import BinTrie

from nlp.NLP import NLPConfig
from nlp.corpus.tag.Nature import Nature
from nlp.corpus.io.IOUtil import loadDictionary 

from nlp.dictionary.CoreDictionary import CoreDictionary

class DynamicCustomDictionary():
    
    # bintrie树
    trie = None
    
    # 双数组trie
    dat = None
    
    # 本词典是从哪些路径加载得到的
    path = []
    
    # 是否执行字符正规化（繁体->简体，全角->半角，大写->小写），切换配置后必须删CustomDictionary.txt.bin缓存
    normalization = NLPConfig.Normalization

    def __init__(self, path):
        self.trie = BinTrie()
        self.dat  = DoubleArrayTrie()

        if path:
            self.load(path)

    def load(self, path) -> bool:
        """加载指定路径的词典"""
        if not path:
            return False
        
        if not self.loadMainDictionary(path[0], path, self.dat, True, self.normalization):
            print(f'自定义词典{path}加载失败')
            return False
        
        print(f"自定义词典加载成功:{self.dat.getSize()}个词条");
        self.path = path

        return True

    def loadMainDictionary(self, mainPath, path, dat, isCache, normalization) -> bool:
        """
        加载词典
        @param mainPath 缓存文件文件名
        @param path     自定义词典
        @param isCache  是否缓存结果
        """
        #print(f'自定义词典开始加载：{mainPath}')
        #if self.loadDat(mainPath, dat):
        #    return True

        try:
            treeMap = {}
            for p in path:
                nature_map = loadDictionary(p, defaultNature = Nature.n)
                treeMap.update(nature_map)
            
            if len(treeMap.keys()) <= 0:
                print('没有加载到任何词条')
                treeMap['other'] = None

            self.dat.build(treeMap)

        except Exception as e:
            print(f'自定义词典{path}加载失败！{e}')

        return True
    
    def insert(self, word, natureWithFrequency) -> bool:
        """
        往自定义词典中插入一个新词（覆盖模式），动态增删不会持久化到词典文件

        @param word                新词 如“裸婚”
        @param natureWithFrequency 词性和其对应的频次，比如“nz 1 v 2”，null时表示“nz 1”。
        @return 是否插入成功（失败的原因可能是natureWithFrequency问题，可以通过调试模式了解原因）
        """
        if not word:
            return False

        att = CoreDictionary.Attribute.create(natureWithFrequency) if natureWithFrequency else CoreDictionary.Attribute(Nature.nz, 1)
        if not att:
            return False

        if self.dat.set(word, attr):
            return True

        self.trie.put(word, att)

        return True
