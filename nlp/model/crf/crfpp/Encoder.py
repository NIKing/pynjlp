import io
import math

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
        @param templFile     模板数据文件
        @param trainFile     训练数据文件
        @param modelFile     模型文件
        @param textModelFile 是否输出文本形式的模型文件
        @param maxitr        最大迭代次数
        @param freq          特征最低频次，用作模型的特征瘦身
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
        
        # open() 作用：
        # 1，读取模版数据，创建一元和二元语法模版对象，并生成模版字符串；
        # 2，读取训练数据，创建标注集合(y)，统计训练数据的长度(xSize)；
        # 3，当然，这里还有检查文件格式是否正确的功能；
        featureIndex = EncoderFeatureIndex(threadNum)
        if not featureIndex.open(templFile, trainFile):
            print('Fail to open [' + templFile + '] and [' + trainFile + ']')
        
        x = []      # 收集taggerImpl 实例化对象，也可以认为是语料库的每个句子实例化对象
        try:
            with open(trainFile, 'r', encoding = 'utf8') as br:
                lineNo = 0
                while True:
                    tagger = TaggerImpl(TaggerImpl.Mode.LEARN)
                    tagger.open(featureIndex)
                    
                    # 注意，这里是按流读取每一行训练数据，数据是按照语料库写入的，会存在空格情况。
                    # status 存在 3 种情况：
                    # 1：错误，当训练数据的gram长度与设置不符，或者训练数据的标签超出设定标签集合长度
                    # 2：文件读取完毕
                    # 3：读取正常，但是无法正常编译特征
                    status = tagger.read(br)
                    if status == TaggerImpl.ReadStatus.ERROR:
                        print('error when reading ' + trainFile)
                        return False
                    
                    # 收集需要瘦身的tagger, 训练文件的每条数据是按照语料库中每条句子进行
                    # not empty() 就是碰到某一个句子读取完毕（碰到了空格情况）
                    if not tagger.empty():
                        
                        # shrink() 作用很多，会调用featureIndex对象的编译特征模版功能
                        # 针对整个句子进行特征编译
                        if not tagger.shrink():
                            print('fail to build feature index')
                            return False

                        tagger.setThread_id(lineNo % threadNum)
                        x.append(tagger)

                    elif status == TaggerImpl.ReadStatus.EOF:
                        print('文件全部读取完毕')
                        break

                    else:
                        continue
                    
                    # 记录训练数据的句子的数量
                    lineNo += 1

                    if lineNo % 100 == 0:
                        print(f'{lineNo}...')


        except OSError as er:
            print('error:', er)
            return False
        
        # 真正瘦身在这里执行
        featureIndex.shrink(freq, x)
        
        # size, 所以特征的标签总数
        # 当前虽然在这里 setAlpha() 了，但是给alpha赋值的地方在 LbfgsOptimizer.lbfgs_optimizer() 中
        alpha = [0.0] * featureIndex.size()
        featureIndex.setAlpha(alpha)

        print(f"Number of sentences: {len(x)}")
        print(f"Number of features:  {featureIndex.size()}")
        print(f"Number of thread(s): {threadNum}")
        print(f"Number of alpha:     {len(alpha)}")
        print(f"Maxitr:              {maxitr}")
        print(f"C:                   {C}")
        print(f"eta:                 {eta}")
        print(f"shrinking size:      {shrinkingSize}")
        print(f"algorithm:           {algorithm}")
        print(f"Freq:                {freq}")

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

        #if not featureIndex.save(modelFile, textModelFile, Encoder.MODEL_VERSION):
        #    print("Failed to save model")
        
        print("Done!")

        return True
    
    def runCRF(self, x, featureIndex, alpha, maxItr, C, eta, shrinkingSize, threadNum, orthant):
        """
        CRF 训练
        - param x             标记列表 
        - param featureIndex  特征索引对象
        - param alpha         最大特征索引组成的数组
        - param maxItr        最大迭代次数
        - param C             损失值
        - param eta           收敛阈值
        - param shrinkingSize 未使用
        - param threadNum     线程数
        - param orthant       是否使用L1范数，默认L2
         return 是否成功
        """
        oldObj = 1e+37
        converge = 0
        
        # 优化器函数
        lbfgs = LbfgsOptimizer()
        
        # 创建多线程
        def create_threads():
            threads = [] * threadNum
            for i in range(threadNum):
                thread = CRFEncoderThread(len(alpha))
                thread.start_i = i
                thread.size = len(x)
                thread.threadNum = threadNum
                thread.x = x

                threads.append(thread)

            return threads

        threads = create_threads()
        
        all_ = 0
        for i in range(len(x)):
            all_ += x[i].size()
        
        print('执行线程')
        print(featureIndex.test_alpha())

        for itr in range(maxItr):
            featureIndex.clear()

            try:
                # 提交线程所有任务
                for thread in threads:
                    thread.start()
                
                # 等待线程执行完毕
                for thread in threads:
                    thread.join()

            except RuntimeError as e:
                print(f'线程错误:{e}')
                return False

            print('-----------------') 
            # 将所有线程数据汇总到第一个线程内
            for i in range(1, threadNum):
                #print(i, threads[i].obj, threads[i].err, threads[i].zeroone)
                threads[0].obj += threads[i].obj
                threads[0].err += threads[i].err
                threads[0].zeroone += threads[i].zeroone
            
            # 期望值
            for i in range(1, threadNum):
                for j in range(featureIndex.size()):
                    threads[0].expected[j] += threads[i].expected[j]

            # 计算非零数量
            numNonZero = 0
             
            if orthant :    # 使用 L1 范数
                for k in range(featureIndex.size()):
                    threads[0].obj += Math.abs(alpha[k] / C)
                    if alpha[k] != 0.0:
                        numNonZero += 1

            else:
                numNonZero = featureIndex.size()
                for k in range(featureIndex.size()):
                    threads[0].obj += (alpha[k] * alpha[k] / (2.0 * C))
                    threads[0].expected[k] += alpha[k] / C
            
            # 尝试释放内存
            for i in range(i, threadNum):
                threads[i].expected = None
            
            try:
                diff = 1.0 if itr == 0 else (abs(oldObj - threads[0].obj) / oldObj)
            except ZeroDivisionError:
                print('ZeroDivsionError')
                diff = float('inf')

            
            print(f"iter:  {itr}/{maxItr}")
            print(f"diff:  {diff}")
            print(f"terr:  {(1.0 * threads[0].err / all_)}")
            print(f"serr:  {(1.0 * threads[0].zeroone / all_)}")
            
            print(f"act:        {numNonZero}")
            print(f"obj:        {threads[0].obj}")
            print(f"expected:   {len(threads[0].expected)}")
            print(f"orthant:    {orthant}")
            print(' ')
            
            oldObj = threads[0].obj
            
            if diff < eta:
                converge += 1
            else:
                converge = 0

            if itr >= maxItr or converge >= 3:
                break
            
            # 在这里会根据损失值和期望值，更新权重到 alpha
            # crf并没有在线学习机制，整个一起进行多线程计算损失值
            ret = lbfgs.optimize(featureIndex.size(), alpha, threads[0].obj, threads[0].expected, orthant, C)
            if ret <= 0:
                return False
            
            # 重置多线程
            threads = create_threads()

        return True


    def runMIRA(self):
        pass

        



