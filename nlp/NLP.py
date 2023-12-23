class NLPConfig:
    
    # 核心词典路径
    CoreDictionaryPath = 'data/dictionary/CoreNatureDictionary.txt'

    # 用户自定义词典路径
    CustomDictionaryPath = ['/pynjlp/data/dictionary/custom/CustomDictionary.txt']
    
    # 是否执行字符正规化（繁体->简体，全角->半角，大写->小写），切换配置后必须删CustomDictionary.txt.bin缓存
    Normalization = False 
