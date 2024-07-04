import re

from decimal import Decimal, ROUND_HALF_UP

from nlp.model.crf.crfpp.FeatureIndex import FeatureIndex

from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger

from nlp.corpus.io.IOUtil import readlinesTxt, writeListToBin, writeTxtByList 
from nlp.utility.Predefine import Predefine

class EncoderFeatureIndex(FeatureIndex):
    def __init__(self, n):
        super().__init__()

        self.threadNum = n
        self.dic = MutableDoubleArrayTrieInteger()
        self.bId = Predefine.INTEGER_MAX_VALUE 

        self.frequency = []
    
    def clear(self):
        pass

    def getID(self, key):
        k = self.dic.get(key)
        
        if k == -1:
            # 在字典没有找到key的情况下，maxid作为字典树中的value
            self.dic.put(key, self.maxid)
            self.frequency.append(1)
            
            # 计算maxid， 这是给下一个key使用
            n = self.maxid
            if key[0] == 'U':
                self.maxid += len(self.y) # key是一元语法/二元语法特征 maxid就是累加标签长度，分词而言等于4
            else:
                self.bId = n    # 注意，bId 只有在 key 是转移特征 的时候 才会重新设置
                self.maxid += len(self.y) * len(self.y) # key是转移特征 maxid就是累加标签长度的平方，分词而言等于16
            
            # 返回与key绑定的value值（maxid）
            return n

        else:
            
            # 计算连续id，不能直接使用k值，因为k值的计算方式的原因，它是不连续的，k值始终比frequency长度要大
            # 如果直接使用 k 值当作 frequency 的索引值，那么 frequency 将会很稀疏，并且不连续，变得难以管理。
            # 因此需要使用某种方法，将不连续的 k 映射到一个连续的范围内，方便对频率的管理
            cid = self.continuousId(k)
            #print('cid', cid)

            oldVal = self.frequency[cid]
            self.frequency[cid] = oldVal + 1

            return k

    def continuousId(self, id):
        """
        获取id在frequency中的位置，此方法的目的是将不连续的id映射到一个连续的范围，确保列表的紧凑型和管理的简便性
        """
        
        # bId用于区分不同范围的id，代表是上一次 发生【转移】的时候 添加key的value值
        if id <= self.bId:          
            return int(id / len(self.y))

        return int(id / len(self.y) - len(self.y) + 1)
        
    def open(self, file_name_1, file_name_2):
        self.checkMaxXsize = True
        return self.openTemplate(file_name_1) and self.openTagSet(file_name_2)

    def openTemplate(self, fileName):
        """读取特征模版文件"""
        isr = readlinesTxt(fileName)

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
            
            # 保存训练数据中最后一条数据的 gram 长度，-1表示去掉标签
            # 因为字符级别的分词，因此 xsize 一直保持为 1
            self.xsize = len(cols) - 1
            
            # 保存训练数据中的标签，去重
            if self.y.count(cols[max_size - 1]) == 0:
                self.y.append(cols[max_size - 1])
        
        return True


    def shrink(self, freq = int, taggers = list):
        """
        收缩(瘦身)
        -param freq 词频
        -param taggers TaggerImpl 对象的集合，指需要被瘦身的特征对象
        """
        
        if freq <= 1:
            return False

        newMaxId = 0
        old2new = defaultdict(int)
        deleteKeys = []
        
        # 读取双数组字典树
        for key, val in self.dic:
            cid = self.continuousId(val)
            f = self.frequency.get(cid)

            if f >= freq:
                old2new[id] = newMaxId
                self.dic[key] = newMaxId
                
                newMaxId += len(self.y) if key[0] == 'U' else len(self.y) * len(self.y)
            
            else:
                deleteKeys.append(key)
        
        # 删除掉低于 freq 的特征，达到瘦身双数组字典树的目的
        for key in deleteKeys:
            self.dic.remove(key)

        for tagger in taggers:
            featureCache = tagger.getFeatureCache()
            #print('featureCache', len(featureCache))

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
        
        self.maxid = newMaxId
    
    def save(self, filename, textModelFile, model_version):
        """
        保存模型
        -paramm filename str 要保存的路径文件名称
        -param textModelFile boolean 是否保存text
        """
        try:
            oos = []
            oos.append(model_version)              
            oos.append(self.costFactor)
            oos.append(self.maxid)
            
            if self.max_xsize > 0:
                self.xsize = min(self.xsize, self.max_xsize)

            oos.append(self.xsize)

            oos.append(self.y)
            oos.append(self.unigramTempls)
            oos.append(self.bigramTempls)
            
            # 双数组字典树, 还不确定如何保存二进制????
            #oos.append(self.dic)
            self.dic.save(oos)

            oos.append(self.alpha)

            # 保存二进制文件
            writeListToBin(filename + '.bin', oos)
            
            if not textModelFile:
                return

            osw = []
            osw.append(f"version:{model_version}")
            osw.append(f"cost-factor:{self.costFactor}")
            osw.append(f"maxid:{self.maxid}")
            osw.append(f"xsize:{self.xsize}")
            osw.append("")
            
            # 标签集合
            for y in self.y:
                osw.append(y)
            osw.append("")
            
            # 一元语法特征模版
            for utempl in self.unigramTempls:
                osw.append(utempl)
            
            # 二元语法特征模版
            for bitempl in self.bigramTempls:
                osw.append(bitempl)
            osw.append("")
            
            # 双字典树
            for pair in self.dic.entrySet():
                osw.append(str(pair.getValue()) + " " + str(pair.getKey()))
            osw.append("")

            for k in range(self.maxid):
                val = Decimal(self.alpha[k]).quantize(Decimal('0.0000000000000000'), rounding=ROUND_HALF_UP)
                osw.append(str(val))
            
            # 保存到文本文件
            writeTxtByList(filename + '.txt', osw)
            
        except Exception as e:
            print(f'Error saving model to {filename}, {e}')

            return False

        return True

