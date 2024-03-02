from nlp.collection.trie.datrie.Utf8CharacterMapping import Utf8CharacterMapping

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
    UNUSED_CHAR_VALUE = UNUSED_CHAR

    def __init__(self, stringIntegerMap = None, entrySet = None, charMap = None):
        self.size = 0
        self.check = []
        self.base  = []

        self.charMap = charMap if charMap != None else Utf8CharacterMapping()

        if stringIntegerMap != None and entrySet == None:
            entrySet = stringIntegerMap.entrySet()

        if entrySet != None:
            for entry in entrySet:
                self.put(entry.getKey(), entry.getValue())

        self.clear()

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

    def isEmpty(self) -> bool:
        return self.size == 0

    def expandArray(self, maxSize):
        """动态扩容数组"""
        curSize = self.getBaseArraySize()

        if curSize > maxSize:
            return

        if maxSize >= self.LEAF_BIT:
            return

        for i in range(maxSize):
            self.base.append(0)
            self.check.append(0)

            self.addFreeLink(i)

    def addFreeLink(self, index):
        """添加自由连接"""
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

        value = self.setLeafValue(value)

        ids = self.charMap.toIdList(key + self.UNUSED_CHAR)
        
        # 根节点（fromState） index 为 1
        fromState, toState, index = 1, 1, 0
        while index < len(ids):
            c = ids[index]
            toState = self.getBase(fromState) + c   # to = base[from] + c
            self.expandArray(toState)

            if self.isEmpty(toState):
                self.deleteFreeLink(toState)

                self.setCheck(toState, fromState) # check[to] = from
                if index == len(ids) - 1:   # Leaf
                    self.size += 1
                    self.setBase(toState, value)
                else:
                    nextChar = ids[(index + 1)]
                    self.setBase(toState, self.getNextFreeBase(nextChar))   # base[to] = free_state - c

            elif self.getCheck(toState) != fromState:
                self.solveConflict(fromState, c)
                continue

            fromState = toState
            index += 1


        if overwrite:
            self.setBase(toState, value)

        return True
    
    def setLeafValue(self, value) -> int:
        """最高4位-置1，或运算"""
        return value | self.LEAF_BIT

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

    def getNextFreeBase(self, nextChar) -> int:
        index = -self.getCheck(0)

        while index != 0:
            # 因为 Root 的index从1开始，所以至少大于1
            if index > nextChar + 1:
                return index - nextChar

            index = -self.getCheck(index)

        oldSize = self.getBaseArraySize()
        self.expandArray(oldSize + 10240)

        return oldSize

    def transfer(self, state, ids):
        """
        转移状态
        -param state
        -param ids/codePoint
        当这个参数类型为数组的时候，循环获取每个元素的状态
        当类型为数字时候，执行toIdList()，再执行调用自身函数获取元素状态
        return -1转移失败
        """

        if state < 1:
            return -1

        if state != 1 and self.isEmpty(state):
            return -1

        if isinstance(ids, list):
            for c in ids:
                if self.getBase(state) + c < self.getBaseArraySize() \
                        and self.getCheck(self.getBase(state) + c) == state:
                    state = self.getBase(state) + c
                else:
                    return -1

            return state

        ids = self.charMap.toIdList(ids)
        if len(ids) == 0:
            return -1

        return self.transfer(state, ids) 

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
        chidlren.remove(newChild)

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
