from nlp.NLP import NLPConfig

from nlp.model.crf.CRFTagger import CRFTagger
from nlp.model.perceptron.PerceptronSegmenter import PerceptronSegmenter

from nlp.dictionary.other.CharTable import CharTable

from nlp.corpus.io.IOUtil import loadInstance

class CRFSegmenter(CRFTagger):
    def __init__(self, modelPath = ""):

        # 不能使用 not modelPath，只有当为默认值的时候，代表可以使用默认模型路径
        if modelPath == "":
            modelPath = NLPConfig.CRFCWSModelPath 

        super().__init__(modelPath)
        
        self.proceptronSegmenter = None
        if modelPath:
            self.proceptronSegmenter = PerceptronSegmenter(this.model)
    
    def segment(self, text, normalized = "", wordList = []):
        if not normalized:
            self.segment(text, CharTable.convert(normalized), wordList)
            return wordList

        self.proceptronSegmenter.segment(text, self.createInstance(normalized), wordList)
    
    def convertCorpus(self, filePath) -> list:
        """转换【BMES】标记"""
        sentences = loadInstance(filePath)
        
        bw = []
        for sentence in sentences:
            
            for word in sentence.toSimpleWordList():
                word = CharTable.convert(word.value)

                if len(word) == 1:
                    bw.append(''.join([word, "\t", "S"]))
                    continue

                bw.append(''.join([word[0], "\t", "B"]))
                
                for c in word[1: len(word) - 1]:
                    bw.append(''.join([c, "\t", "M"]))
                
                bw.append(''.join([word[-1], "\t", "E"]))
        
            bw.append("\t")

        return bw

    def getDefaultTemplateData(self):
        return ['# Unigram', 'U0:%x[-1,0]', 'U1:%x[0,0]', 'U2:%x[1,0]', 'U3:%x[-2,0]%x[-1,0]', 'U4:%x[-1,0]%x[0,0]', 'U5:%x[0,0]%x[1,0]', 'U6:%x[1,0]%x[2,0]', '# Bigram', 'B']


