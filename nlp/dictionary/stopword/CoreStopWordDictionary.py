from nlp.NLP import NLPConfig
from nlp.corpus.io.IOUtil import readlinesTxt

class CoreStopWordDictionary():
    dictionary = set()
    
    @staticmethod
    def load(coreStopWordDictionaryPath, loadCacheIfPossible):
        CoreStopWordDictionary.dictionary = StopWordDictionary(coreStopWordDictionarypath)
    
    @staticmethod
    def reload():
        allWords = readlinesTxt(NLPConfig.CoreStopWordDictionaryPath)
        
        print('', len(allWords))
        for word in allWords:
            CoreStopWordDictionary.dictionary.add(word)


    @staticmethod
    def contains(key = str) -> bool:
        return key in CoreStopWordDictionary.dictionary
