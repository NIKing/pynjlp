import re

from nlp.model.crf.crfpp.FeatureIndex import FeatureIndex

from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger

from nlp.corpus.io.IOUtil import readlinesTxt 
from nlp.utility.Predefine import Predefine

class EncoderFeatureIndex(FeatureIndex):
    def __init__(self, n):
        super().__init__()

        self.threadNum = n
        self.dic = MutableDoubleArrayTrieInteger()
        self.bId = Predefine.INTEGER_MAX_VALUE 

        self.frequency = []
    
    def getID(self, key):
        k = self.dic.get(key)
        if k == -1:
            self.dic.put(key, self.maxid)
            self.frequency.append(1)
            
            n = self.maxid
            if key[0] == 'U':
                self.maxid += len(self.y)
            else:
                bId = n
                self.maxid += len(self.y) * len(self.y)
            
            return n

        else:

            cid = self.continousId(k)
            #print('cid', cid)

            oldVal = self.frequency[cid]
            self.frequency[cid] = oldVal + 1

            return k

    def continousId(self, id):
        if id <= self.bId:
            return int(id / len(self.y))

        return int(id / len(self.y) - len(self.y) + 1)
        
    def open(self, file_name_1, file_name_2):
        self.checkMaxXsize = True
        return self.openTemplate(file_name_1) and self.openTagSet(file_name_2)

    def openTemplate(self, fileName):
        """读取特征模版文件"""
        isr = readlinesTxt(fileName)
        print(f'有{len(isr)}条模版数据')

        for line in isr:
            if len(line) == 0 or line[0] == ' ' or line[0] == '#':
                continue
            
            # 收集一元语法模型和二元语法模型
            if line[0] == 'U':
                self.unigramTempls.append(line.strip())
            elif line[0] == 'B':
                self.bigramTempls.append(line.strip())
            else:
                print('system error unknow type: ' + line)

        self.templs = self.makeTempls(self.unigramTempls, self.bigramTempls)

        return True

    def openTagSet(self, fileName):
        """读取训练文件中的标签集"""
        max_size = 0
        isr = readlinesTxt(fileName)
        print(f'有{len(isr)}条标签数据')

        self.y.clear()

        for line in isr:
            if len(line) == 0:
                continue

            firstChar = line[0]
            if firstChar == '\0' or firstChar == ' ' or firstChar == '\t':
                continue
            
            cols = re.split('[\t ]', line)
            if max_size == 0:
                max_size = len(cols)

            if max_size != len(cols):
                print('inconsistent column size ' + fileName)
            
            # 获取语料库中代表单词的最大数量，-1表示去掉标签
            self.xsize = len(cols) - 1
            
            # 收集语料库中的标签，去重
            if self.y.count(cols[max_size - 1]) == 0:
                self.y.append(cols[max_size - 1])
        
        return True


    def shrink(self, freq = int, taggers = list):
        """
        收缩(瘦身)
        -param freq 词频
        -param taggers 标签集
        """
        
        if freq <= 1:
            return False

        newMaxId = 0
        old2new = defaultdict(int)
        deleteKeys = []

        for key, val in self.dic:
            cid = self.continuousId(val)
            f = self.frequency.get(cid)

            if f >= freq:
                old2new[id] = newMaxId
                self.dic[key] = newMaxId
                newMaxId += len(self.y) if key[0] == 'U' else len(self.y) * len(self.y)
            else:
                deleteKeys.append(key)

        for key in deleteKeys:
            self.dic.remove(key)


        for tagger in taggers:
            featureCache = tagger.getFeatureCache()

            for k in range(len(featureCache)):
                featureCacheItem = featureCache.get(k)
                newCache = []

                for it in featureCacheItem:
                    if it == -1:
                        continue

                    nid = old2new.get(it)
                    if not nid:
                        newCache.append(nid)

                newCache.append(-1)
                featureCache.set(k, newCache)
        
        self.maxid = newMaxid
            
