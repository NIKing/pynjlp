from nlp.model.perceptron.feature.FeatureMap import FeatureMap
from nlp.collection.trie.datrie.MutableDoubleArrayTrieInteger import MutableDoubleArrayTrieInteger 

"""用MutableDoubleArrayTrie实现的ImmutableFeatureMap"""
class ImmutableFeatureMDatMap(FeatureMap):
    def __init__(self, dat = None, tagSet = None, featureIdMap = None, featureIdSet = None):
        super().__init__(tagSet)
        
        if dat != None:
            self.dat = dat
            return
       
        self.dat = MutableDoubleArrayTrieInteger(featureIdMap)
        
        if featureIdSet != None:
            for entry in featureIdSet:
                self.dat.put(entry.getKey(), entry.getValue())
    
    def __len__(self):
        return self.dat.size

    def idOf(self, string):
        return self.dat.get(string)

    def getSize(self) -> int:
        return self.dat.size

    def entrySet(self):
        return self.dat.entrySet()

    def save(self, out):
        """保存模型参数"""
        
        # 保存标签分类
        self.tagSet.save(out)
        
        # 保存双数组字典树
        self.dat.save(out)

    def load(self, byteArray):
        """加载模型"""

        # 加载分类标签
        super().loadTagSet(byteArray)

        # 加载双数组字典树
        return self.dat.load(byteArray)

