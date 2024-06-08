from nlp.model.crf.CRFTagger import CRFTagger

class CRFSegmenter(CRFTagger):
    def __init__(self, modelPath = ""):
        if not modelPath:
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




