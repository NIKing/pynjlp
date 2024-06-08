from nlp.model.crf.LogLinearModel import LogLinearModel

from nlp.model.crf.crfpp.Encoder import Encoder
from nlp.model.crf.crfpp.crf_learn import crf_learn

class CRFTagger():
    def __init__(self, modelPath):
        if not modelPath:
            return 

        self.model = LogLinearModel(modelPath)

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
            algorithm = crf_learn.algorithm

        encoder = Encoder()
        if not encoder.learn(tempFile, trainFile, modelFile,
                True, maxitr, freq, eta, cost, trhead, shrinking_size,
                algorithm):
            raise ValueError('fail to learn model')


