import sys
sys.path.append('/pynjlp')

from pyhanlp import *

HanLP = JClass('com.hankcs.hanlp.HanLP')
ClusterAnalyzerByHanLP = JClass('com.hankcs.hanlp.mining.cluster.ClusterAnalyzer')
CoreStopWordDictionary = JClass('com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary')

#HanLP.Config.enableDebug()
segment = HanLP.newSegment()
def preprocess(word):
    termList = segment.seg(word)
    #print(termList)

    wordList = []
    for term in termList:
        term = term.toString()
        word, nature = term.split('/')[0],''.join(term.split('/')[1:])
        if CoreStopWordDictionary.contains(word) or nature.startswith("w"):
            continue

        wordList.append(word)

    print(wordList)

from nlp.mining.cluster.ClusterAnalyzer import ClusterAnalyzer

def analyzer_by_myself():
    analyzer = ClusterAnalyzer()
    words_list = [
        ["肝脏", "未", "分化", "肉瘤", "胚胎", "型"],
        ["nk", "t", "细胞", "淋巴瘤"],
        ["肾", "细胞", "癌"],
        ["急性", "髓", "细胞", "白血病"],
        ["胚胎", "性", "横纹肌", "肉瘤"],
        ["伯基特", "淋巴瘤"],
        ["胶质", "神经元", "混合性", "肿瘤"],
        ["纵膈", "精原细胞", "瘤"],
        ["骶", "尾部", "恶性肿瘤"],
        ["毛细胞", "型", "星形", "细胞瘤"],
        ["双眼", "视网膜", "母细胞瘤"],
        ["经典", "霍奇金", "淋巴瘤"]
    ]

    for i, words in enumerate(words_list):
        analyzer.addDocument(i, words)

    #res = analyzer.repeatedBisection(limit_eval = 1.0)
    res = analyzer.kmeans(3)
    print('res', res)

def analyzer_by_hanlp():
    analyzer = ClusterAnalyzerByHanLP()
    words = ['肝脏未分化肉瘤胚胎型', 'nk/t细胞淋巴瘤', '肾细胞癌', '急性髓细胞白血病', '胚胎性横纹肌肉瘤', '伯基特淋巴瘤', '胶质神经元混合性肿瘤', '纵膈精原细胞瘤', '骶尾部恶性肿瘤', '毛细胞型星形细胞瘤', '双眼视网膜母细胞瘤', '经典霍奇金淋巴瘤']
    for i, word in enumerate(words):
        #preprocess(word)
        analyzer.addDocument(i, word)

    
    res = analyzer.repeatedBisection(1.0)
    print('res', res)


if __name__ == "__main__":
    
    analyzer_by_myself()
    #for i in range(10):
        #analyzer_by_myself()
        #analyzer_by_hanlp()
        #print(' ') 





