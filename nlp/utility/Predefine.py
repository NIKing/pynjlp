
class Predefine:
    
    # 最小int值
    INTEGER_MIN_VALUE = -2147483648

    # 最小概率值
    MIN_PROBABILITY = 1e-10
    
    WORD_SEGMENTER = '@'

    # 现在总词频25146057
    TOTAL_FREQUENCY = 25146057
    
    # 未登录词的默认词频
    OOV_DEFAULT_FREQUENCY = 10000

    # Unigram 平滑因子
    _lambda = 0.9
    
    # Bigram 平滑因子
    myu = 1 - 1 / TOTAL_FREQUENCY + 0.00001

    # 开始标记
    TAG_BEGIN = '始##始'

    # 结束标记
    TAG_END = '末##末'
    
    # 数词 m
    TAG_NUMBER = '未##数'

    # 地址 ns
    TAG_PLACE = '未##地'
    
    # 其它
    TAG_OTHER = "未##它"
    
    # 团体名词 nt
    TAG_GROUP = "未##团"
    
    # 数量词 mq
    TAG_QUANTIFIER = "未##量"
    
    # 专有名词 nx
    TAG_PROPER = "未##专"
    
    # 时间 t
    TAG_TIME = "未##时"
    
    # 字符串 x
    TAG_CLUSTER = "未##串"
    
    # 人名 nr
    AG_PEOPLE = "未##人"
    
    # 二进制文件后缀
    BIN_EXT = ".bin"
    
    @staticmethod
    def setTotalFrequency(totalFrequency):
        Predefine.TOTAL_FREQUENCY = totalFrequency
        Predefine.myu = 1 - ( 1 / totalFrequency + 0.00001 )
        Predefine.OOV_DEFAULT_FREQUENCY = max(1, min(Predefine.OOV_DEFAULT_FREQUENCY / 100, totalFrequency))
