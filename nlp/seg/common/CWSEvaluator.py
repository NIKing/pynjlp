from nlp.corpus.io.IOUtil import lineIterator, readlinesTxt, writeTxtByList
import re

"""中文分词评测工具"""
class CWSEvaluator:

    def __init__(self, dic):
        self.dic = set()

        self.A_size = 0
        self.B_size = 0
        
        self.A_cap_B_size = 0

        self.IV_R = 0
        self.OOV_R = 0

        self.IV = 0
        self.OOV = 0

        if isinstance(dic, set):
            self.dic = dic
            return

        if not dic:
            return

        wordList = readlinesTxt(dic)
        for word in wordList:
            if not word:
                continue
             
            self.dic.add(word)

    
    def compare(self, gold, pred):
        """比较标准答案和预测结果"""
        wordArray = re.split(r"\s+", gold)
        predArray = re.split(r"\s+", pred)

        self.A_size += len(wordArray)
        self.B_size += len(predArray)
        
        goldIndex, predIndex = 0, 0
        goldLen, predLen = 0, 0

        while goldIndex < len(wordArray) and predIndex < len(predArray):
            if goldLen == predLen:
                if wordArray[goldIndex] == predArray[predIndex]:
                    if self.dic:
                        if wordArray[goldIndex] in self.dic:
                            self.IV_R += 1
                        else:
                            self.OOV_R += 1

                    self.A_cap_B_size += 1

                goldLen += len(wordArray[goldIndex])
                predLen += len(predArray[predIndex])

                goldIndex += 1
                predIndex += 1

            elif goldLen < predLen:
                goldLen += len(wordArray[goldIndex])
                goldIndex += 1
            else:
                predLen += len(predArray[predIndex])
                predIndex += 1
        
        # 统计新词和登录词数量
        if self.dic:
            for word in wordArray:
                if word in self.dic:
                    self.IV += 1
                else:
                    self.OOV += 1

    def getResult(self, percentage = True):
        """
        获取PRF值
        -param percentage 是否百分制
        """

        p = self.A_cap_B_size / self.B_size
        r = self.A_cap_B_size / self.A_size

        if percentage:
            p *= 100
            r *= 100

        oov_r = float('nan')
        if self.OOV > 0:
            oov_r = self.OOV_R / self.OOV
            if percentage:
                oov_r *= 100
        
        iv_r = float('nan')
        if self.IV > 0:
            iv_r = self.IV_R / self.IV
            if percentage:
                iv_r *= 100

        return (p, r, 2 * p * r / (p + r), oov_r, iv_r)
    

    @staticmethod
    def calculate(goldFile, predFile, dictPath):
        """
        计算结果
        -param goldFile 标准答案
        -param prefFile 预测结果
        -param dictPath 所有词字典路径
        """
        goldIter = lineIterator(goldFile)
        predIter = lineIterator(predFile)

        evaluator = CWSEvaluator(dictPath)

        gold = next(goldIter)
        pred = next(predIter)
        
        # 通过对比标准答案和预测结果，进行模型评测
        while gold and pred:
            evaluator.compare(gold, pred)

            gold = next(goldIter, None)
            pred = next(predIter, None)

        return evaluator.getResult()
    
    @staticmethod
    def evaluate(segment, outputPath, goldFile, dictPath):
        """
        标准化评测分词器
        -param segment 分词器
        -param outputPath 分词预测输出文件
        -param goldFile 测试集 segmented file
        -param dictPath 训练集单词列表
        return 一个存储准确率的结构
        """
        lineIterators = readlinesTxt(goldFile)
        
        # 分词后(预测结果)，写入文件内
        wordList = []
        for line in lineIterators:
            line = re.sub(r"\s+", "", line)
            words = segment.seg(line)
            
            wordList.append(''.join([word + "  " for word in words]).rstrip("  "))
        
        writeTxtByList(outputPath, wordList)
        
        return CWSEvaluator.calculate(goldFile, outputPath, dictPath)
