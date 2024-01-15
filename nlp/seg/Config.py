class Config():
    
    # 是否是索引分词（合理地最小分割），indexMode代表全切分词语的最小长度（包含）
    indexMode = 0
    
    # 是否加载用户词典
    useCustomDictionary = False
    
    # 词性标注
    speechTagging = False

    # 是否识别中国人名
    nameRecognize = True

    # 是否识别音译人名
    translatedNameRecognize = True

    # 是否识别日本人名
    japaneseNameRecognize = False

    # 是否识别地名
    placeRecognize = False

    # 是否识别机构
    organizationRecognize = False

    # 命名实体识别是否至少有一项被激活
    ner = True
    
    #是否计算偏移量
    offset = False

    def updateNerConfig(self):
        self.ner = self.nameRecognize or self.translatedNameRecognize or self.japaneseNameRecognize or self.placeRecognize or self.organizationRecognize
