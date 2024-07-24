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
        """这里不仅构建特征空间(dic)，还为每个特征构建词频（frequency）"""
        k = self.dic.get(key)
        
        if k == -1:
            # 在字典没有找到key的情况下，maxid作为字典树中的value
            self.dic.put(key, self.maxid)
            
            # 注意，这里并没有用 maxid 当作 frequency 数组的索引，而我们需要把 maxid 和 frequency 的索引建立关系
            # 因此，在下面处理不是新特征的时候，通过 continousId() 函数 获取 maxid 与 frequency 的对应关系
            self.frequency.append(1)
            
            # 计算maxid， 这是给下一个特征当作值
            # 当模版是一元/二元语法特征的时候('U')，maxid += 标记长度，标记长度根据训练数据获取，当前默认是4
            # 当模版是转移特征('B')，maxid += 标记长度 * 2
            n = self.maxid
            if key[0] == 'U':
                self.maxid += len(self.y) # key是一元语法/二元语法特征 
            else:
                self.bId = n    # 注意，bId 只有在 key 是转移特征 的时候 才会重新设置
                self.maxid += len(self.y) * len(self.y) # key是转移特征 maxid就是累加标签长度的平方，分词而言等于16
            
            # 返回与key绑定的value值（maxid）
            return n

        else:
            
            # 计算连续id，不能直接使用k值，因为k值是不连续的，k值始终比frequency长度要大
            # 如果直接使用 k 值当作 frequency 的索引值，那么 frequency 将会很稀疏，并且不连续，变得难以管理。
            # 因此需要使用某种方法，将不连续的 k 映射到一个连续的范围内，方便对频率的管理
            cid = self.continuousId(k)
            
            self.frequency[cid] += 1

            return k

    def continuousId(self, id):
        """
        获取id在frequency中的位置，此方法的目的是将不连续的id映射到一个连续的范围，确保列表的紧凑型和管理的简便性
        """
        
        # 不太理解，为何会有这样的判断？一直觉得 bId 会保持增长，当 id 是之前转移特征编号的话，岂不是找错了位置
        if id <= self.bId:          
            return int(id / len(self.y))
        
        # 什么情况下会调用到它呢？当上一个特征模版是 B 的时候，而当前 id 就会小于 bId，而调用它的目的就是为了保证，在frequency中当前id也必须紧挨着上一个id
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
        
        # 整个语法模型字符串
        self.templs = self.makeTempls(self.unigramTempls, self.bigramTempls)

        return True

    def openTagSet(self, fileName):
        """读取训练文件中的标签集"""
        max_size = 0
        isr = readlinesTxt(fileName)
        #print(f'有{len(isr)}条标签数据')

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
            # 在 TaggerImpl 中需要使用 xsize 来获取训练数据的标签位置 
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
        
        # 读取双数组字典树（特征空间）, 所有训练数据的特征值都在这里
        for key, val in self.dic:
            cid = self.continuousId(val)
            f = self.frequency.get(cid)

            if f >= freq:
                old2new[id] = newMaxId
                self.dic[key] = newMaxId
                
                newMaxId += len(self.y) if key[0] == 'U' else len(self.y) * len(self.y)
            
            else:
                deleteKeys.append(key)
        
        # 删除掉低于 freq 的特征，达到瘦身双数组字典树(特征空间)的目的
        for key in deleteKeys:
            self.dic.remove(key)
        
        # 在构建特征空间的时候，同时，根据特征编号在句子的标记类中构建了特征缓存
        # 因此，需要对标记类进行特征缓存的瘦身
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
            #print('双数组字典树长度', len(self.dic))
            for pair in self.dic.entrySet():
                osw.append(str(pair.getValue()) + " " + str(pair.getKey()))
            osw.append("")
            
            # 权重 Decimal().quantize() 将 self.alpha[k] 的值转换为浮点数，并将其量化为 16 位小数点数值，后面不够位数就自动补0
            for k in range(self.maxid):
                val = Decimal(self.alpha[k]).quantize(Decimal('0.0000000000000000'), rounding=ROUND_HALF_UP)
                osw.append(str(val))
            
            # 保存到文本文件
            writeTxtByList(filename + '.txt', osw)
            
        except Exception as e:
            print(f'Error saving model to {filename}, {e}')

            return False

        return True

