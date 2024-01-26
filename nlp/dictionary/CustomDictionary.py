from nlp.dictionary.DynamicCustomDictionary import DynamicCustomDictionary
from nlp.NLP import NLPConfig

class CustomDictionary():
    
    # 默认实例
    DEFAULT = DynamicCustomDictionary(NLPConfig.CustomDictionaryPath)

