import time

from abc import ABC, abstractmethod

from nlp.model.crf.LogLinearModel import LogLinearModel

from nlp.model.crf.crfpp.Encoder import Encoder, Algorithm
from nlp.model.crf.crfpp.crf_learn import crf_learn

from nlp.corpus.io.IOUtil import writeTxtByList

class CRFTagger(ABC):
    def __init__(self, modelPath):
        if not modelPath:       # 训练模式, 也就是说训练的时候并没有用到线性模型
            return 
        
        self.model = LogLinearModel(modelFile = modelPath)

    @abstractmethod
    def convertCorpus(self, filePath):
        pass

    def train(self, trainFile, modelFile, templFile = "",
            maxitr = 10000, freq = 1, eta = 0.0001, C = 1.0, threadNum = 1, 
            shrinkingSize = 20, algorithm = None):
        """
        @param templFile     模板文件
        @param trainFile     训练文件
        @param modelFile     模型文件
        @param maxitr        最大迭代次数
        @param freq          特征最低频次
        @param eta           收敛阈值
        @param C             cost-factor
        @param threadNum     线程数
        @param shrinkingSize
        @param algorithm     训练算法
        @return
        """

        if algorithm == None:
            maxitr  = crf_learn.maxitr
            freq    = crf_learn.freq
            eta     = crf_learn.eta
            cost    = crf_learn.cost
            thread  = crf_learn.thread

            shrinking_size = crf_learn.shrinking_size
            algorithm = Algorithm.fromString(crf_learn.algorithm)
        
        # 需要在生成模版文件的同时，根据训练数据生成标签集【B,M,E,S】训练数据
        if not templFile:
            # 生产模版数据
            templateData = self.getDefaultTemplateData()

            fileName  = 'crfpp-template-' + time.strftime("%Y-%m-%d", time.localtime()) + '.txt'
            templFile = '/pynjlp/data/test/crf-cws-model/' + fileName

            writeTxtByList(templFile, templateData)

            # 生成标签集数据
            trainData = self.convertCorpus(trainFile)
            trainData = trainData[:10000]

            trainFileName = 'crfpp-train-' + time.strftime("%Y-%m-%d", time.localtime()) + '.txt'
            trainFile = '/pynjlp/data/test/crf-cws-model/' + trainFileName

            writeTxtByList(trainFile, trainData)
        
        encoder = Encoder()
        if not encoder.learn(templFile, trainFile, modelFile,
                True, maxitr, freq, eta, cost, thread, shrinking_size,
                algorithm):
            
            raise ValueError('fail to learn model')





