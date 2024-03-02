from nlp.model.perceptron.feature.ImmutableFeatureMDatMap import ImmutableFeatureMDatMap

"""不可变的特征映射"""
class LockableFeatureMap(ImmutableFeatureMDatMap):
    def __init__(self, tagSet):
        super().__init__(tagSet = tagSet)

    def idOf(self, string) -> int:
        id = super().idOf(string)
        
        # 当双数组可变，就插入键值
        if id == -1 and self.mutable:
            id = self.dat.getSize()
            self.dat.put(string, id)

        return id
