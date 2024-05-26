from nlp.model.perceptron.feature.FeatureMap import FeatureMap
from nlp.model.perceptron.feature.DefaultDictWithEntrySet import DefaultDictEntrySet

class MutableFeatureMap(FeatureMap):
    def __init__(self, tagSet):
        super().__init__(tagSet, True)

        self.featureIdMap = DefaultDictEntrySet(int)
        self.addTransitionFeature(tagSet)
   
    def getSize(self) -> int:
        return len(self.featureIdMap)
    
    def addTransitionFeature(self, tagSet):
        """
        添加训练特征，给标签添加前缀字符
        -param tagSet 默认标签集{B,E,M,S}
        """
        for tag in tagSet:
            self.idOf("BL=" + tag)
        
        self.idOf("BL=_BL_")

    def idOf(self, key):
        val = self.featureIdMap.get(key)
        if not val:
            val = len(self.featureIdMap)
        
        self.featureIdMap[key] = val 

        return val
       

        

        
