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

    def size(self) -> int:
        return self.dat.size

    def entrySet(self):
        return self.dat.entrySet()

    def save(self, out):
        self.tagSet.save(out)
        self.dat.save(out)
