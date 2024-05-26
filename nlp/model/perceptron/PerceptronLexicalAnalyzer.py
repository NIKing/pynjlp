from nlp.tokenizer.lexical.AbstractLexicalAnalyzer import AbstractLexicalAnalyzer
from nlp.model.perceptron.model.LinearModel import LinearModel

from nlp.model.perceptron.PerceptronSegmenter import PerceptronSegmenter
from nlp.model.perceptron.PerceptronPOSTagger import PerceptronPOSTagger
from nlp.model.perceptron.PerceptronNERecognizer import PerceptronNERecognizer

"""感知机词法分析器"""
class PerceptronLexicalAnalyzer(AbstractLexicalAnalyzer):
    def __init__(self, cwsModelFile = "", posModelFile = "", nerModelFile = ""):
        self.segmenter = None
        self.posTagger = None
        self.neRecognizer = None
        
        cwsModel, posModel, nerModel = None, None, None
        
        # 初始化分词模型
        if isinstance(cwsModelFile, str) and cwsModelFile != "":
            cwsModel = LinearModel(modelFile = cwsModelFile)
        elif cwsModelFile:
            cwsModel = cwsModelFile
        
        if cwsModel != None:
            self.segmenter = PerceptronSegmenter(cwsModel)
        
        # 初始化词性模型
        if isinstance(posModelFile, str) and posModelFile != "":
            posModel = LinearModel(modelFile = posModelFile)
        elif posModelFile:
            posModel = posModelFile

        if posModel != None:
            self.posTagger = PerceptronPOSTagger(posModel)
            self.config.speechTagging = True
        
        # 初始化命名实体识别模型
        if isinstance(nerModelFile, str) and nerModelFile != "":
            nerModel = LinearModel(modelFile = nerModelFile)
        elif nerModelFile:
            nerModel = nerModelFile
        
        if nerModel != None:
            self.neRecognizer = PerceptronNERecognizer(nerModel)
            self.config.ner = True

        super().__init__(self.segmenter, self.posTagger, self.neRecognizer)


    def segment(self, text):
        if not text:
            return ""


