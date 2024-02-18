class NLPConfig:
    
    # 核心词典路径
    CoreDictionaryPath = '/pynjlp/data/dictionary/CoreNatureDictionary.mini.txt'
    
    # 二元语法词典路径
    BiGramDictionaryPath = "/pynjlp/data/dictionary/CoreNatureDictionary.ngram.txt"

    # 用户自定义词典路径
    CustomDictionaryPath = ['/pynjlp/data/dictionary/custom/CustomDictionary.txt']
    
    # 是否执行字符正规化（繁体->简体，全角->半角，大写->小写），切换配置后必须删CustomDictionary.txt.bin缓存
    Normalization = False

    # 字符类型对应表
    CharTypePath = "/pynjlp/data/dictionary/other/CharType.bin"

    # 字符正规化表（全角转半角，繁体转简体）
    CharTablePath = "/pynjlp/data/dictionary/other/CharTable.txt"

    # 词性标注集描述表，用来进行中英映射（对于Nature词性，可直接参考Nature.java中的注释）
    PartOfSpeechTagDictionary = "/pynjlp/data/dictionary/other/TagPKU98.csv"
