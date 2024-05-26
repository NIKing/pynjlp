from nlp.model.perceptron.feature.DefaultDictWithEntrySet import DefaultDictEntrySet
from nlp.model.perceptron.feature.FeatureMap import FeatureMap

class ImmutableFeatureMap(FeatureMap):
    def __init__(self, featureIdMap, entrySet = None, tagSet = None):
        super().__init__(tagSet)
        
        if featureIdMap:
            self.featureIdMap = featureIdMap
            return
        
        self.featureIdMap = DefaultDictEntrySet(int)

        if not entrySet:
            return

        for entry in entrySet:
            self.featureIdMap[entry.getKey()] = entry.getValue()

    def idOf(self, key):
        id = self.featureIdMap.get(key)
        if not id:
            return -1

        return id
    
    def getSize(self):
        return len(self.featureIdMap)

    def entrySet(self):
        return self.featureIdMap.entrySet()

