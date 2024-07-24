class Instance():
    def __init__(self):
        self.tagArray = []
        self.featureMatrix = []
        self.sentence = []

    def __len__(self):
        return len(self.tagArray)

    def addFeature(self, rawFeature, featureVector, featureMap):
        """
        添加特征到特征向量
        -param rawFeature 未经过加工的特征
        -param featureVector 特征向量
        -param featureMap 特征映射对象
        return void
        """
        id = featureMap.idOf(''.join(rawFeature))
        
        if id != -1:
            featureVector.append(id)
    
    def toFeatureArray(self, featureVector) -> list:
        """
        特征向量转换数组
        """
        # 添加一列，给转移特征
        featureArray = [0] * (len(featureVector) + 1)
        for i, feature in enumerate(featureVector):
            featureArray[i] = feature

        return featureArray

    def size(self):
        return len(self.featureMatrix)

    def getFeatureAt(self, position):
        return self.featureMatrix[position]

    @staticmethod
    def addFeatureThenClear(rawFeature, featureVector, featureMap):
        """添加特征，同时清空缓存"""
        id = featureMap.idOf(''.join([str(r) for r in rawFeature]))
        if id != -1:
            featureVector.append(id)

        rawFeature = []


