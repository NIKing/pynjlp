from nlp.collection.trie.bintrie.HashCode import hash_code, char_hash
from nlp.collection.trie.DoubleArrayTrieSearcher import Searcher, LongestSearcher

class Node:
    def __init__(self):
        self.c = ''
        self.code = 0
        self.depth = 0
        self.left = 0
        self.right = 0

    def toString(self):
        return "Node{ code="+ self.code +" ,depth="+ self.depth +",left="+ self.left +",right="+ self.right +"}"

class DoubleArrayTrie:
    check = []
    base  = []
    
    size = 0
    allocSize = 0

    key = []
    value = []
    length = []

    char = []
    v = []
    
    progress = 0
    nextCheckPos = 0  # 下一次检查的位置

    def __init__(self, buildFrom = None, enableFastBuild = False):

        self.size = 0
        self.keySize = 0
        self.allocSize = 0
        self.error = 0
        
        self.enableFastBuild = enableFastBuild 

        if buildFrom and self.build(buildFrom) != 0:
            print("构造失败")
    
    def getSize(self):
        return self.size

    def resize(self, newSize):
        """拓展数组"""
        base2  = [0] * newSize
        check2 = [0] * newSize
        char2  = [0] * newSize

        if self.allocSize > 0:
            base2   = self.base[:self.allocSize]
            check2  = self.check[:self.allocSize]
            char2   = self.char[:self.allocSize]
        
        self.base  = base2
        self.check = check2
        self.char  = char2
        
        self.allocSize = newSize

    def build(self, keyValueMap):
        """构建DAT，注意keys一定要进行字符排序，否则会构建失败"""
        assert keyValueMap != None
        keyValueList = [(k, keyValueMap[k]) for k in sorted(keyValueMap.keys())]
        
        keys, values = [], []
        for key, val in keyValueList:
            if not key:
                continue

            keys.append(key)
            values.append(val)
        
        return self.build_items(list(keys), list(values))

    def build_items(self, keyList, valueList):
        assert len(keyList) == len(valueList), "键的个数与值的个数不一样！"
        assert len(keyList) > 0 , "键值个数为0！"
        
        self.v = valueList

        return self.build_keys(keyList, None, None, len(keyList))

    def build_keys(self, _key, _length, _value, _keySize):
        """
        -param list _key 必须是字典序
        -param list _length 对应每个key的长度，留空动态获取
        -param list _value 每个key对应的值，留空使用key的下标作为值
        -param int key的长度，应该设为_key, size
        return 是否出错
        """

        if not _key or _keySize > len(_key):
            return 0
        
        self.key = _key
        self.length = _length
        self.keySize = _keySize
        self.value = _value

        self.progress = 0
        self.allocSize = 0

        # 32个双字节
        self.resize(65536 * 32)

        self.base[0] = 1
        self.nextCheckPos = 0

        
        # 根节点
        root_node = Node()
        root_node.left  = 0
        root_node.right = _keySize
        root_node.depth = 0

        siblings = self.fetch(root_node)
        self.insert(siblings, {})
        self.shrink()
        
        self.key = None
        self.length = None

        return self.error

    def fetch(self, parent) -> list:
        """
        获取直接相连的子节点, 没有就创建节点
        -param parent 父节点
        -return 兄弟节点列表
        """
        if self.error < 0:
            return []

        #print('parent____c=', parent.c, 'right===', parent.right, 'left===', parent.left, 'code===', parent.code)
        
        siblings = []
        prev, i = 0, parent.left
        while i < parent.right:

            length = self.length[i] if self.length != None else len(self.key[i])
            if length < parent.depth:
                i += 1
                continue

            tmp = self.key[i]

            cur = 0
            length = self.length[i] if self.length != None else len(tmp)
            if length != parent.depth:
                # 在java中使用 (int) 获取字符的Unicode值，在这里使用hash_code()获取
                cur = char_hash(tmp[parent.depth]) + 1
            
            if prev > cur:
                self.error = -3
                return []

            if cur != prev or len(siblings) == 0:
                tmp_node = Node()
                tmp_node.depth = parent.depth + 1
                tmp_node.code  = cur
                tmp_node.left  = i
                tmp_node.c     = tmp[parent.depth] if len(tmp) > parent.depth else 0
                
                if len(siblings) != 0:
                    siblings[len(siblings) - 1].right = i

                siblings.append(tmp_node)

            prev = cur
            i += 1

        if len(siblings) != 0:
            siblings[len(siblings) - 1].right = parent.right
        
        #print('siblings====', [(s.c, s.code, s.left, s.right) for s in siblings])
        return siblings
    
    def insert(self, siblings, used):
        """
        插入节点
        -param list siblings 等待插入的兄弟节点
        -return int 插入位置
        """
        if self.error < 0:
            return 0
        
        # pos 来自当前兄弟节点中第一个节点的hash值或父节点最后
        begin = 0
        pos = max(siblings[0].code + 1, (self.nextCheckPos + 1) if self.enableFastBuild else self.nextCheckPos) - 1
        nonzero_num = 0
        first = 0

        if self.allocSize <= pos:
            self.resize(pos + 1)
        
        #print(f'====insert-Start====【pos = {pos}】 ==== 【char = {siblings[0].c}】===【code={siblings[0].code}】')

        """
        扩容策略
        此循环的目标是找出满足check[begin + a1...an] == 0 的 n 个空闲空间, a1...an是siblings的n个节点
        """
        while True:
            pos += 1
            
            if self.allocSize <= pos:
                self.resize(pos + 1)
            
            if self.check[pos] != 0:
                nonzero_num += 1
                continue

            if first == 0:
                self.nextCheckPos = pos
                first = 1
            
            # 当前位置离第一个兄弟节点的距离, 因为经过排序，所以只需要根据第一个节点计算距离即可
            # pos 可能是兄弟节点第一个字符的hash值，也有可能是父节点最后一个字符hash值
            begin = pos - siblings[0].code 
            
            # 保证起始位置开始后面有足够的空间，为下面的循环做铺垫
            if self.allocSize <= (begin + siblings[len(siblings) - 1].code):
                self.resize(begin + siblings[len(siblings) - 1].code + 65535)
            
            if used.get(begin, 0) == 1:
                continue
            
            # 这种循环方式是为了实现 java 中的 continue outer；当内循环正常终止，才终止外循环，否则继续外循环
            # 这句话的意思，如果找到了满足目标空闲空间，就跳出整个循环
            for i in range(1, len(siblings)):
                if self.check[begin + siblings[i].code] != 0:
                    break
            else:
                break

        """
        启发式空闲搜索法
        从位置nextCheckPos开始到pos之间，如果占用率在95%以上，下次插入节点时，直接从pos位置开始查找 
        """ 
        if 1.0 * nonzero_num / (pos - self.nextCheckPos + 1) >= 0.95:
            self.nextCheckPos = pos

        used.setdefault(begin, 1)

        if self.size <= begin + siblings[len(siblings) - 1].code + 1:
            self.size = begin + siblings[len(siblings) - 1].code + 1

        # 检查每个子节点 
        # 给 check 赋值，建立子节点 (check[begin + s.code]) 与父节点 (begin) 的多对一的关系
        # check的对应位置是：字符hash值 + 1 ，其赋的值是：上一个字符的位置(父节点)，初始等于1
        for i in range(len(siblings)):
            self.check[begin + siblings[i].code] = begin
            self.char[begin + siblings[i].code] = siblings[i].c
            #print(f'====insert-Middle=【begin={begin}】【({i}), char={siblings[i].c} code={siblings[i].code}】【pos={pos}】')
        
        # 检查每个子节点, 若其没有孩子，就将它的base设置 -1，否则就调用 insert 建立关系 
        # 给 base 赋值，建立父节点 (base[begin + s.code]) 与子节点 (h) 的一对多的关系
        for i in range(len(siblings)):
            new_siblings = self.fetch(siblings[i])
            
            # 一个词的终止且不为其他词的前缀
            if len(new_siblings) == 0:
                # 会给终止符设置负值
                self.base[begin + siblings[i].code] = (-self.value[siblings[i].left] - 1) if self.value else (-siblings[i].left - 1)
                
                if self.value and (-self.value[i].left - 1) >= 0:
                    self.error = 2
                    return 0

                self.progress += 1

            else:
               h = self.insert(new_siblings, used)
               self.base[begin + siblings[i].code] = h

        return begin

    def shrink(self):
        """释放空闲的内存"""
        nbase = [0] * (self.size + 65535)
        nbase = self.base[:self.size]

        self.base = nbase

        ncheck = [0] * (self.size + 65535)
        ncheck = self.check[:self.size]

        self.check = ncheck

        nchar = [0] * (self.size + 65535)
        nchar = self.char[:self.size]

        self.char = nchar

    def toString(self):
        infoIndex   = "i    = "
        infoChar    = "char = "
        infoBase    = "base = "
        infoCheck   = "check= "
        
        for i in range(len(self.base)):
            if self.base[i] == 0 and self.check[i] == 0:
                continue
            
            infoChar    += "   " + str(self.char[i])
            infoIndex   += " " + str(i)
            infoBase    += " " + str(self.base[i])
            infoCheck   += " " + str(self.check[i])


        return "DoubleArrayTrie{" + \
                "\n" + infoChar + \
                "\n" + infoIndex + \
                "\n" + infoBase + \
                "\n" + infoCheck + "\n" + \
                "}"
    
    def exactMatchSearch(self, keyChars, pos = 0, length = 0, nodePos = 0):
        """精确匹配"""
        if length == 0:
            length = len(keyChars)

        if nodePos <= 0:
            nodePos = 0

        result = -1

        b = self.base[nodePos]
        p = 0

        for i in range(pos, length):
            p = b + char_hash(keyChars[i]) + 1
            if b == self.check[p]:
                b = self.base[p]
            else:
                return result
        
        # 走到这里，说明 keyChars 在字典中是一个完整的字符串
        p = b
        n = self.base[p]
        if b == self.check[p] and n < 0:
            result = -n - 1

        return result


    def get(self, key):
        """精确查找"""
        index = self.exactMatchSearch(key)
        if index >= 0:
            return self.v[index]

        return None

    def parseText(self, txt):
        searcher = Searcher(self, 0, txt)

        wordList = []
        while searcher.next():
            
            begin = searcher.begin
            end = begin + searcher.length
            #print(begin, end, searcher.value)
            #print(f'--------{begin}----{end}-----{searcher.value}----------------------')
            #print(' ')

            wordList.append(txt[begin:end])

        return wordList

    
    def parseLongestText(self, txt):
        searcher = LongestSearcher(self, 0, txt)
        
        wordList = []
        while searcher.next():
            begin = searcher.begin
            end = begin + searcher.length
            #print(begin, end, searcher.value)

            wordList.append(txt[begin:end])

        return wordList

    def getLongestSearcher(self, txt, offset):
        return LongestSearcher(self, offset, txt)

    def getSearcher(self, txt, offset):
        return Searcher(self, offset, txt)


