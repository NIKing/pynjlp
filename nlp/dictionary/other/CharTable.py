from nlp.corpus.io.IOUtil import lineIterator, readlinesTxt
from nlp.NLP import NLPConfig 

"""字符表，字符正规化"""
class CharTable:

    CONVERT = []

    MAP = {}

    @staticmethod
    def load(path = ''):
        """加载字符表，主要根据path加载文件，建立字符大写和小写，繁体和简体的映射关系"""
        #print(chr(100), '=', ord('d'), ord('D'))
        #print(ord(chr(0x10FFFF)), 0x10FFFF + 1)
        
        # 列举所有unicode字符
        CONVERT = [chr(c) for c in range(0, 0x10FFFF + 1)]
        CharTable.CONVERT = [chr(c) for c in range(0, 0x10FFFF + 1)]

        #print(CharTable.CONVERT[:10])
        
        # 这个转换有问题，在字符表中，出现之前字符被替换，后面还使用的情况，这会报错
        #print('"' in CharTable.CONVERT, ord('"'))
        #print('“' in CharTable.CONVERT, ord('”'))

        # 加载自定义字符表，大小写和繁简体字符
        lineList = readlinesTxt(path)

        for line in lineList:
            # 文件内的数据格式：A=a，啢=唡
            if len(line) != 3:
                continue
            
            #if line[0] not in CharTable.CONVERT:
            #    print('没找到的字符', line[0], line[0] in CONVERT)
            #    continue

            #if line[2] not in CharTable.CONVERT:
            #    print('没找到的字符', line[2], line[2] in CONVERT)
            #    continue

            #print(line)
            #print(CharTable.CONVERT.index(line[0]), CharTable.CONVERT.index(line[2]))
            #print(line[0], chr(ord(line[0])), line[2])

            if line[0] == '，':
                print(CharTable.CONVERT.index(line[0]))
                print(CONVERT[CONVERT.index(line[2])])
            
            CharTable.CONVERT[CharTable.CONVERT.index(line[0])] = CONVERT[CONVERT.index(line[2])]
        
        # 设置空格
        CharTable.loadSpace()
    
    @staticmethod
    def loadSpace():
       for i in range(0, 0x10FFFF + 1):
           if ord(' ') == i:
               CharTable.CONVERT[i] = ' '

    @staticmethod 
    def load_by_map(path = ''):
        # 把所有unicode字符放在映射表商
        for c in range(0, 0x10FFFF + 1):
            c = chr(c)
            CharTable.MAP[c] = c

        # 加载自定义字符表，大小写和繁简体字符
        lineList = readlinesTxt(path)
        
        for line in lineList:
            # 文件内的数据格式：A=a，啢=唡
            if len(line) != 3:
                continue
            
            CharTable.MAP[line[0]] = line[2]
    
    @staticmethod
    def convert_by_str(string) -> list:
        if len(string) == 1:
            return CharTable.MAP[string]

        return ''.join([CharTable.MAP[s] for s in string])

    @staticmethod
    def convert(string) -> list:
        if isinstance(string, str):
            return CharTable.convert_by_str(string)
        
        return []

    @staticmethod
    def normalization(sentence = str) -> str:
        if len(sentence) <= 0:
            return sentence
        
        sentences = CharTable.convert(sentence)
        return ''.join(sentences)
        



CharTable.load_by_map(NLPConfig.CharTablePath)
