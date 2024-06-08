from enum import Enum, auto

class Algorithm(Enum):
    CRF_L1 = auto()
    CRF_L2 = auto()
    MIRA   = auto()
    
    @staticmethod
    def fromString(algorithm):
        algorithm = algorithm.lower()

        if algorithm == "crf" or algorithm == "crf-l2":
            return Algorithm.CRF_L2
        elif algorithm == "crf-l1":
            return Algorithm.CRF_L1
        elif algorithm == "mira":
            return Algorithm.MIRA

class Encoder():
    MODEL_VERSION = 100

    def learn(self, templFile, trainFile, modelFile, textModelFile,
            maxitr, freq, eta, C, threadNum, shrinkingSize,
            algorithm):
        """
        训练
        @param templFile     模板文件
        @param trainFile     训练文件
        @param modelFile     模型文件
        @param textModelFile 是否输出文本形式的模型文件
        @param maxitr        最大迭代次数
        @param freq          特征最低频次
        @param eta           收敛阈值
        @param C             cost-factor
        @param threadNum     线程数
        @param shrinkingSize
        @param algorithm     训练算法
        @return
        """
        if eta <= 0:
            print('eta must be > 0.0')
            return False

        if C < 0.0:
            print('C must be >= 0.0')
            return False

        if shrinkingSize < 1:
            print('shrinkingSize must be >= 1')
            return False

        if threadNum <= 0:
            print('thread must be > 0')
            return False

        featureIndex = EncoderFeatureIndex(threadNum)
        if featureIndex.open(templFile, trainFile):
            print('Fail to open' + templFile + " " + trainFile)
        
        x = []
        try:
            br = IOUtil.readText(trainFile)

            lineNo = 0
            while True:
                tagger = TaggerImpl(TaggerImpl.Mode.LEARN)
                tagger.open(featureIndex)

                status = tagger.read(br)
                if status == TaggerImpl.ReadStatus.ERROR:
                    print('error when reading ' + trainFile)
                    return False

                if not tagger.empty():
                    if not tagger.shrink():
                        print('fail to build feature index')
                        return False

                    tagger.setThread_id_(lineNo % threadNum)
                    x.append(tagger)

                elif status == TaggerImpl.ReadStatus.EOF:
                    break

                else:
                    continue

                if lineNo % 100 == 0:
                    print(lineNo + '.. ')

        except:
            return False

        featureIndex.shrink(freq, x)

        alpha = [0.0] * len(featureIndex.size())
        featureIndex.setAlpha_(alpha)

        print("Number of sentences: " + len(x))
        print("Number of features:  " + featureIndex.size())
        print("Number of thread(s): " + threadNum)
        print("Freq:                " + freq)
        print("eta:                 " + eta)
        print("C:                   " + C)
        print("shrinking size:      " + shrinkingSize)

        if algorithm == Algorithm.CRF_L1 and \
            not self.runCRF(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum, True):
                print("CRF_L1 execute error")
                return False

        if algorithm == Algorithm.CRF_L2 and \
            not self.runCRF(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum, False):
                print("CRF_L2 excute error")
                return False
        
        if algorithm == Algorithm.MIRA and \
            not self.runMIRA(x, featureIndex, alpha, maxitr, C, eta, shrinkingSize, threadNum):
                print("MIRA excute error")
                return False

        if not featureIndex.save(modelFile, textModelFile):
            print("Failed to save model")

        print("Done!")

        return True
    
    def runCRF(self, x, featureIndex, alpha, maxItr, C, eta, shrinkingSize, threadNum, orthant):
        """
        CRF 训练
        - param x             句子列表
        - param featureIndex  特征编号表
        - param alpha         特征函数的代价
        - param maxItr        最大迭代次数
        - param C             cost factor
        - param eta           收敛阈值
        - param shrinkingSize 未使用
        - param threadNum     线程数
        - param orthant       是否使用L1范数
         return 是否成功
        """
        oldObj = 1e+37
        converge = 0

        lbfgs = LbfgsOptimizer()
        
        #threads = []
        #for i in range(threadNum):
        #    thread = CRFEncoderThread(alpha.length)
        #    thread.start_i = i
        
        all_ = 0
        for i in range(len(x)):
            all_ += x[i].size()

        executor = Executors.newFixedThreadPool(threadNum)
        for itr in range(maxItr):
            featureIndex.clear()

            try:
                executor.invokeAll(threads)
            except:
                return False



    def runMIRA(self):
        pass

        



