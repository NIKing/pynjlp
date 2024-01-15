from nlp.utility.Predefine import Predefine
from nlp.dictionary.CoreBiGramTableDictionary import CoreBiGramTableDictionary

import math

class MathUtility():
    
    @staticmethod
    def calculateWeight(_from, _to) -> float:
        """
        计算两个节点之间的权重(二元语法概率, 可利用 极大似然概率 + 平滑策略 得到)，注意两个节点具有前后关系
        原注释：从一个词到另一个词的词的花费

        -param _from Vertex对象 前节点
        -param _to Vertex对象 后节点
        return 分数
        """

        # 获取前后节点在一元语法模型中统计的词频
        fFrom   = _from.getAttribute().totalFrequency  
        fTo     = _to.getAttribute().totalFrequency

        # 计算两个节点在二元语法模型中共同出现的词频
        fBigram = CoreBiGramTableDictionary.getBiFrequency(_from.wordID, _to.wordID)
        
        #print(f'{_from.realWord} === {_from.word}')
        #print(f'{_to.realWord} === {_to.word}')
        #print(f'{_from.wordID}')
        #print(f'{_to.wordID}')
        #print(f'from={fFrom}, to={fTo}，fBigram={fBigram}')

        # _lambda 和 myu 属于两个平滑因子，值属于 (0, 1)
        # TOTAL_FREQUENCY 属于语料库总频次
        l = Predefine._lambda * (Predefine.myu * fBigram / (fFrom + 1) + 1 - Predefine.myu) 
        r = (1 - Predefine._lambda) * fTo / Predefine.TOTAL_FREQUENCY
        
        return -math.log(l + r)

