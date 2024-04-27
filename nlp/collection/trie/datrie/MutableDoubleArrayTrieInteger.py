from nlp.collection.trie.datrie.Utf8CharacterMapping import Utf8CharacterMapping
from nlp.collection.trie.datrie.SetMap import SetMap
from nlp.collection.trie.datrie.KeyValuePair import KeyValuePair

"""
可变的双数组trie树
"""
class MutableDoubleArrayTrieInteger:
    
    serialVersionUID = '5586394930559218802L'

    LEAF_BIT = 1073741824

    EMPTY_WALK_STATE = {-1, -1}
    
    # 字符串的终止字符（会在传入的字符串末尾添加该字符）
    UNUSED_CHAR = '\000'
    
    # 终止字符的codePoint, 这个字符作为叶节点的标识
    # ord() 既可以把字符转unicode码点，也可以把 ascii 码转 unicode码
    UNUSED_CHAR_VALUE = ord(UNUSED_CHAR)

    def __init__(self, stringIntegerMap = None, entrySet = None, charMap = None):
        self.size = 0
        self.check = []
        self.base  = []

        self.charMap = charMap if charMap != None else Utf8CharacterMapping()

        self.clear()

        if stringIntegerMap != None and entrySet == None:
            entrySet = stringIntegerMap.entrySet()

        if entrySet != None:
            for entry in entrySet:
                self.put(entry.getKey(), entry.getValue())


    def entrySet(self):
        pair = KeyValuePair(self)
        
        for s in range(self.size):
            yield pair.next() 

    def clear(self):
        self.base  = []
        self.check = [] 

        self.base.append(0)
        self.check.append(0)
        
        self.base.append(1)
        self.check.append(0)

        self.expandArray(self.charMap.getInitSize())

    def getSize(self) -> int:
        """键值对个数"""
        return self.size

    def isSizeEmpty(self) -> bool:
        return self.size == 0

    def expandArray(self, maxSize):
        """动态扩容数组"""
        curSize = self.getBaseArraySize()

        if curSize > maxSize:
            return

        if maxSize >= self.LEAF_BIT:
            return

        for i in range(curSize, maxSize + 1):
            self.base.append(0)
            self.check.append(0)

            self.addFreeLink(i)

    def addFreeLink(self, index):
        """添加自由连接"""
        # self.base[0] 是上一个节点，获取上一个节点在 check[] 数组中的数据，作为当前节点在 check[] 数组中的数据
        self.check[index] = self.check[-self.base[0]]
        self.check[-self.base[0]] = -index

        self.base[index] = self.base[0]
        self.base[0] = -index

    def deleteFreeLink(self, index):
        """将index从空闲循环链表中删除"""
        self.base[-self.check[index]] = self.base[index]
        self.check[-self.base[index]] = self.check[index]

    def insert(self, key, value, overwrite = False):
        """
        插入条目
        -param key       键
        -param value     值
        -param overwrite 是否覆盖
        return
        """
        if key == None or len(key) == 0 or self.UNUSED_CHAR in key:
            return False

        if value < 0 or (value & self.LEAF_BIT) != 0:
            return False
        
        #print('insert', key, value)
        value = self.setLeafValue(value)
        ids = self.charMap.toIdList(key + self.UNUSED_CHAR)     # 一串字符将变成UTF-8类型的字符数组, eg: 珍 = b'\xe7\x8f\x8d\x00'
        #print('change:', value, ids, f'ids_0={ids[0]}')
        
        # 根节点（fromState = 1)
        fromState, toState, index = 1, 1, 0
        while index < len(ids):
            c = ids[index]
            toState = self.getBase(fromState) + c   # to = base[from] + c
            self.expandArray(toState)
            
            #print(f'b={fromState}')
            #print(f'begin={self.getBase(fromState)}', f'c={c}')
            #print(f'p={toState}')
            #print(f'check={self.getCheck(toState)}')
            
            # 前后字符没有建立父子关系
            if self.isEmpty(toState):
                self.deleteFreeLink(toState)
                
                # 建立转移成功条件, 给当前节点建立父
                self.setCheck(toState, fromState) # check[to] = from
                #print(f'to_check={toState}, value={fromState}')
                
                # 最后一个子节点，设置值
                if index == len(ids) - 1:   
                    self.size += 1          # 代表一组字典添加成功
                    self.setBase(toState, value)
                    #print(f'to_base={toState}, value={value}')
                else:
                    # 建立父子关系, 把子节点挂在当前当节点上
                    nextChar = ids[(index + 1)]
                    self.setBase(toState, self.getNextFreeBase(nextChar))   # base[to] = free_state - c
                    #print(f'to_base={toState}, value={self.getNextFreeBase(nextChar)}')
            
            # 若建立的关系与父级不同，则进行修正
            elif self.getCheck(toState) != fromState:
                #print('solveConf')
                self.solveConflict(fromState, c)
                continue

            fromState = toState
            index += 1
            
            #print('----------')
        
        # 覆盖最后转移字符的值，不管是否字符串中最后一个字符（默认最后一个字符应该是 000)
        if overwrite:
            self.setBase(toState, value)

        return True
    
    def isLeafValue(self, value):
        return value > 0 and value & self.LEAF_BIT != 0

    def setLeafValue(self, value) -> int:
        """最高4位置1，'|' 使用二进制，按位或运算"""
        return value | self.LEAF_BIT

    def getLeafValue(self, value) -> int:
        """最高4位置0， '^'  使用二进制，按位异或运算, 相当于把或运算还原"""
        return value ^ self.LEAF_BIT

    def getBaseArraySize(self) -> int:
        return len(self.base)

    def getBase(self, index) -> int:
        return self.base[index]

    def getCheck(self, index) -> int:
        return self.check[index]

    def setBase(self, index, value):
        self.base[index] = value

    def setCheck(self, index, value):
        self.check[index] = value

    def isEmpty(self, index):
        return self.getCheck(index) <= 0

    def transfer(self, state, ids):
        """
        转移状态
        -param state
        -param ids/codePoint
        当这个参数类型为数组的时候，循环获取每个元素的状态
        当类型为数字时候，执行toIdList()，再执行调用自身函数获取元素状态
        return -1 转移失败
        """

        if state < 1:
            return -1

        if state != 1 and self.isEmpty(state):
            return -1

        if isinstance(ids, list):
            for c in ids:
                toState = self.getBase(state) + c
                if toState < self.getBaseArraySize() and self.getCheck(toState) == state:
                    state = toState
                else:
                    return -1

            return state
        
        # 当 ids 参数实际上是 int 类型的 codePoint 的时候
        ids = self.charMap.toIdList(ids)
        if len(ids) == 0:
            return -1
    
        # 自调用
        return self.transfer(state, ids)


    def stateValue(self, state):
        leaf = self.getBase(state) + self.UNUSED_CHAR_VALUE
        if self.getCheck(leaf) == state:
            return self.getLeafValue(self.getBase(leaf))

        return -1

    def get(self, key, start = 0) -> int:
        """
        精确查找
        -param key 
        -param start 查找的起始位置
        return -1 表示不存在
        """
        assert key != None
        assert 0 <= start and start <= len(key)
        
        state = 1
        ids = self.charMap.toIdList(key[start:])
        state = self.transfer(state, ids)

        if state < 0:
            return -1

        return self.stateValue(state)

    def set(self, key, value) -> bool:
        """
        设置键值（同put）
        -param key
        -param value
        return 是否设置成功
        """
        return self.insert(key, value, True)

    def put(self, key, value) -> bool:
        """
        设置键值（同set）
        -param key
        -param value
        return 是否设置成功
        """
        return self.insert(key, value, True)

    def add(self, key, value) -> bool:
        """
        非覆盖模式添加
        -param key
        -param value
        return 是否设置成功
        """
        return self.insert(key, value, False)

    def remove(self, key) -> int:
        return self.delete(key)

    def delete(self, key):
        if not key:
            return -1

        curState = 1
        ids  = self.charMap.toIdList(key)
        path = [0] * (len(ids) + 1)

        for i in range(len(ids)):
            c = ids[i]
            if self.getBase(curState) + c >= self.getBaseArraySize() \
                    or (self.getCheck(self.getBase(curState) + c) != curState):
                        break

            curState = self.getBase(curState) + c


        ret = -1
        if i == len(ids):
            if self.getCheck(self.getBase(curState) + self.UNUSED_CHAR_VALUE) == curState:
                this.size -= 1

                ret = self.getLeafValue(self.getBase(self.getBase(curState)) + self.UNUSED_CHAR_VALUE)
                path[(len(path)-1)] = self.getBase(curState) + self.UNUSED_CHAR_VALUE

                for j in range(len(path) - 1, -1, -1):
                    isLeaf = True
                    state = path[i]

                    for k in range(self.charMap.getCharsetSize()):
                        if self.isLeafValue(self.getBase(state)):
                            break

                        if self.getBase(state) + k < self.getBaseArraySize() \
                                and self.getCheck(self.getBase(state)) + k == state:
                                    isLeaf = False
                                    break

                    if not isLeaf:
                        break

                    self.addFreeLink(state)

        return ret

    
    def solveConflict(self, parent, newChild):
        """
        解决冲突
        -param parent 父节点
        -param newChild 字节点的chr值
        """

        # 找出parent节点下所有子节点
        children = [newChild]

        charsetSize = self.charMap.getCharsetSize()
        for c in range(charsetSize):
            _next = self.getBase(parent) + c
            if _next >= self.getBaseArraySize():
                break

            if self.getCheck(_next) == parent:
                children.append(c)


        # 移动旧子节点到新的位置
        newBase = self.searchFreeBase(children)
        children.remove(newChild)

        for c in children:
            child = newBase + c
            self.deleteFreeLink(child)

            self.setCheck(child, parent)
            childBase = self.getBase(self.getBase(parent) + c)
            self.setBase(child, childBase)

            if not self.isLeafValue(childBase):
                for d in range(charsetSize):
                    to = childBase + d
                    if to >= self.getBaseArraySize():
                        break

                    if self.getCheck(to) == self.getBase(parent) + c:
                        self.setCheck(to, child)

            self.addFreeLink(self.getBase(parent) + c)

        self.setBase(parent, newBase)

    def searchFreeBase(self, children):
        """寻找空闲空间"""
        minChild = children[0]
        maxChild = children[-1]

        current = 0
        while self.getCheck(current) != 0:
            if current > minChild + 1:
                base = current - minChild
                ok = True

                for it in children:
                    to = base + it
                    if to >= self.getBaseArraySize():
                        ok = False
                        break

                    if not self.isEmpty(to):
                        ok = False
                        break


                if ok:
                    return base

            current = -self.getCheck(current)

        oldSize = self.getBaseArraySize()
        self.expandArray(oldSize + maxChild)

        return oldSize

    def getNextFreeBase(self, nextChar) -> int:
        """为字符找空闲位置在Base数组中"""
        index = -self.getCheck(0)

        while index != 0:
            # 因为 Root 的index从1开始，所以至少大于1
            if index > nextChar + 1:
                return index - nextChar

            index = -self.getCheck(index)

        oldSize = self.getBaseArraySize()
        self.expandArray(oldSize + 10240)

        return oldSize

    def save(self, out):
        """保存模型参数"""
        
        # 保存键值对的数量
        out.append(self.size)

        # 保存双数组的长度
        out.append(self.getBaseArraySize())
        
        # 保存双数组
        out.extend(self.base)
        out.extend(self.check)

    def load(self, byteArray):
        """加载模型"""
        
        self.size = byteArray.next()
        
        # 获取双双数组长度
        array_size = byteArray.next()

        self.base = byteArray.next(count = array_size)
        self.check = byteArray.next(count = array_size) 
   
