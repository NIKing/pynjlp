# 内部类 - 双数组搜索工具 
from .bintrie.HashCode import hash_code, char_hash

class Searcher():
    
    # key的起点
    begin = 0

    # key的长度
    length = 0
    
    # key的字典序坐标
    index = 0

    # key对应的value
    value = ''
    
    # 传入的字符数组
    charArray = []
    
    # charArray的长度
    arrayLenght = 0

    # 上一个节点位置
    last = None
    
    # 上一个字符下标
    i = 0

    def __init__(self, trie, offset, charArray):
        
        self.trie  = trie
        self.base  = trie.base
        self.check = trie.check
        self.v     = trie.v

        self.charArray = charArray
        self.arrayLength = len(charArray)
        
        #print(charArray, self.base[0], self.arrayLength)
        self.i = offset
        self.last = self.base[0]
        
        if self.arrayLength == 0:
            self.begin = -1
        else:
            self.begin = offset

    def next(self) -> bool:
        b = self.last
        n, p = 0, 0

        while True:
            
            #print(f'==【i={self.i}】==【arrayLength={self.arrayLength}】')
            # 指针到头，将起点向前挪一位，重新开始，状态归零
            if self.i == self.arrayLength:
                self.begin += 1

                if self.begin == self.arrayLength:
                    break

                self.i = self.begin
                b = self.base[0]
            
            # 转移状态 p = base[b] + c = base[char[i - 1]] + char[i] + 1
            # 转移成功 base[char[i-1]] == check[base[char[i-1]] + char[i] + 1]
            #print(f'第{self.i}个字符 ({self.charArray[self.i]})，hashcode = {char_hash(self.charArray[self.i]) + 1}')
            p = b + char_hash(self.charArray[self.i]) + 1
            #print(f'<<<<<<<【b={b}】【check[p]={self.check[p]}】【p = {p}】')
            if b == self.check[p]:
                b = self.base[p]
            else:
                # 转移失败 起点向前挪一个，重新开始，状态归零
                # 2024年1月1日，添加了“+1 ”操作，当转移失败，需要从第二个字符开始扫描，不能从头开始
                self.i = self.begin + 1
                
                self.begin += 1
                if self.begin == self.arrayLength:
                    break

                b = self.base[0]
                #print(' ')
                continue
            
            
            p = b
            n = self.base[p]
            #print('------', b, self.check[p], n)
            # 判断是否是终止节点
            if b == self.check[p] and n < 0:
                self.length = self.i - self.begin + 1
                self.index  = -n - 1
                self.value  = self.v[self.index]
                
                self.last = b
                self.i += 1
                
                #print(' ')
                return True

            self.i += 1
        
        #print(f'最后i=【{self.i}】')
        return False


class LongestSearcher():

    # key的起点
    begin = 0

    # key的长度
    length = 0
    
    # key的字典序坐标
    index = 0
    
    # key对应的value
    value = ''
    
    # 传入的字符数组
    charArray = []
    
    # charArray的长度
    arrayLenght = 0

    # 上一个字符下标
    i = 0

    def __init__(self, trie, offset, charArray):
        
        self.base  = trie.base
        self.check = trie.check
        self.v     = trie.v

        self.charArray = charArray
        self.arrayLength = len(charArray)
        
        #print(charArray, self.arrayLength, self.base[0])
        self.i = offset
        self.begin = offset

    def next(self) -> bool:
        self.length = 0
        self.begin  = self.i

        b = self.base[0]
        n, p = 0, 0

        while True:

            # 指针到头，将起点向前挪一个，重新开始，状态归零
            if self.i >= self.arrayLength:
                return self.length > 0
            
            # 状态转移 p = base[char[i-1]] + char[i] + 1  
            # 或者 p = base[b] + c，满足 base[b] = check[p]
            #print(f'第{self.i}个字符 ({self.charArray[self.i]})，hashcode = {char_hash(self.charArray[self.i]) + 1}')
            p = b + char_hash(self.charArray[self.i]) + 1
            #print(f'<<<<<<<【b={b}】【p = {p}】【check[p]={self.check[p]}】')
            
            if b == self.check[p]:
                b = self.base[p] # 转移成功，沿着节点向下走
                #print(f'++++++转移成功【b = {b}】')
                
            else:

                if self.begin == self.arrayLength:
                    break
                
                # 输出最长词后，从该词语的下一个位置恢复扫描
                if self.length > 0:
                    self.i = self.begin + self.length
                    return True
                
                # 转移失败，也将起点往前挪一个，重新开始，状态归零
                self.i = self.begin

                # 记住失败的位置，若再次失败，则从这里重新开始
                self.begin += 1
                
                # 状态归零
                b = self.base[0]
                #print(f'-----转移失败【b={b}】【base={self.base[b]}】【check={self.check[b]}】')

            p = b
            n = self.base[p]
            
            # n 的结果小于0，代表到了结束字符 \0 
            if b == self.check[p] and n < 0:
                self.length = self.i - self.begin + 1
                self.index  = -n - 1
                self.value  = self.v[self.index]
            
            self.i += 1

        return False
        
