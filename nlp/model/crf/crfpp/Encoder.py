import io

from enum import Enum, auto

from nlp.model.crf.crfpp.EncoderFeatureIndex import EncoderFeatureIndex
from nlp.model.crf.crfpp.TaggerImpl import TaggerImpl

from nlp.model.crf.crfpp.LbfgsOptimizer import LbfgsOptimizer
from nlp.model.crf.crfpp.CRFEncoderThread import CRFEncoderThread

from nlp.corpus.io.IOUtil import readlinesTxt 

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

"""编码器"""
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
        
        # open() 具有提取语料库特征（好像是一元和二元语法模版)
        # 当然，这里还有检查文件格式是否正确的功能
        featureIndex = EncoderFeatureIndex(threadNum)
        if not featureIndex.open(templFile, trainFile):
            print('Fail to open [' + templFile + '] and [' + trainFile + ']')
        
        x = []
        try:
            with open(trainFile, 'r', encoding = 'utf8') as br:
                lineNo = 0
                while True:
                    tagger = TaggerImpl(TaggerImpl.Mode.LEARN)
                    tagger.open(featureIndex)
                    
                    status = tagger.read(br)
                    if status == TaggerImpl.ReadStatus.ERROR:
                        print('error when reading ' + trainFile)
                        return False
                    
                    #print(status)
                    # 收集需要收缩的tagger
                    if not tagger.empty():
                        if not tagger.shrink():
                            print('fail to build feature index')
                            return False

                        tagger.setThread_id(lineNo % threadNum)
                        x.append(tagger)

                    elif status == TaggerImpl.ReadStatus.EOF:
                        break

                    else:
                        continue
                    
                    lineNo += 1

                    if lineNo % 100 == 0:
                        print(f'{lineNo}...')


        except OSError as er:
            print('error:', er)
            return False
        print('------------')

        # 真正瘦身在这里执行
        featureIndex.shrink(freq, x)

        alpha = [0.0] * len(featureIndex.size())
        featureIndex.setAlpha(alpha)

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
        - param x             标记列表 
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
        
        # 创建多线程
        def create_threads():
            threads = [] * threadNum
            for i in range(threadNum):
                thread = CRFEncoderThread(alpha.length)
                thread.start_i = i
                thread.size = len(x)
                thread.threadNum = threadNum
                thread.x = x

                threads.append(thread)

        threads = create_threads()
        
        all_ = 0
        for i in range(len(x)):
            all_ += x[i].size()
        
        for itr in range(maxItr):
            featureIndex.clear()

            try:
                # 一次性提交线程所有任务
                for thread in threads:
                    thread.start()
                
                # 等待线程执行完毕
                for thread in threads:
                    thread.join()

            except:
                print('线程错误')
                return False
        
            # 将所有线程数据汇总到第一个线程内
            for i in range(1, threadNum):
                threads[0].obj += threads[i].obj
                threads[0].err += threads[i].err
                threads[0].zeroone += threads[i].zeroone
            
            # 期望值
            for i in range(1, threadNum):
                for j in range(len(featureIndex)):
                    threads[0].expected[j] += threads[i].expected[j]

            # 计算非零数量
            numNonZero = 0
            if orthant :
                for k in range(len(featureIndex)):
                    threads[0].obj += Math.abs(alpha[k] / C)
                    if alpha[k] != 0.0:
                        numNonZero += 1

            else:
                numNonZero = len(featureIndex)
                for k in range(len(featureIndex)):
                    threads[0].obj += (alpha[k] * alpha[k] / (2.0 * C))
                    threads[0].expected[k] += alpha[k] / C
            
            # 尝试释放内存
            for i in range(i, threadNum):
                threads[i].expected = None
            
            diff = 1.0 if itr == 0 else (math.abs(oldObj - threads[0].obj) / oldObj)
            print("iter:  " + itr)
            print("terr:  " + (1.0 * threas[0].err / all_))
            print("serr:  " + (1.0 * threas[0].zeroone / all_))
            print("act:   " + numNonZero)
            print("obj:   " + threads[0].obj)
            print("diff:  " + diff)
            
            oldObj = threads[0].obj
            if diff < eta:
                converge += 1
            else:
                converge = 0

            if itr >= maxItr or converge >= 3:
                break

            ret = lbfgs.optimize(len(featureIndex), alpha, threads[0].obj, threads[0].expected, orthant, C)

            if ret <= 0:
                return False
            
            # 重置多线程
            threads = create_threads()

        return True


    def runMIRA(self):
        pass

        



