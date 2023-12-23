from collections import deque
from abc import ABC, abstractmethod

from numpy as np

from nlp.colletion.AhoCorasick.State import State

class Builder(ABC):
    """构建工具"""

    # 根节点，仅仅用于构建过程
    rootState = State()

    # 是否占用，仅仅用于构建
    used = []

    # 已分配在内存中的大小
    allocSize = 0

    # 一个控制增长速度的变量
    progress = 0

    # 下一个插入的位置将从此开始搜索
    nextCheckPos = 0

    # 键值对的大小
    keySize = 0
    
    # fail表
    fail = []
    
    # 输出表
    output = []

    # 双数组值check
    check = []
    
    # base 和 check 的大小
    size = 0

    # 双数组之base
    base  = []

    # 保存value
    v = []
    
    # 每个key的长度
    l = []

    def build(self, treeMap):
        """treeMap是排序好"""
        assert treeMap != None
        treeMap = [(k, treeMap[k]) for k in sorted(treeMap.keys())]
        
        # 保存value
        self.v = treeMap.values()
        
        # 每个key的长度
        self.l = len(self.v)

        keySet = treeMap.keys()

        # 构建二分树, 普通字典树
        self.addAllKeyword(keySet)

        # 构建双数组树
        self.buildDoubleArrayTrie(keySet)
        self.used = None
        
        # 构建failure表
        self.constructFailureStates()
        self.rootState = None
        self.loseWeight()

    def addAllKeyword(self, keySet):
        for i, keyword in enumerate(keySet):
            self.addKeyword(keyword, i)
    
    def addKeyword(self, keyword, index):
        """添加一个键值"""

        # 构建 goto 表，字典树
        currentState = self.rootState
        for character in keyword:
            currentState = currentState.addState(character)
        
        # 构建 output 表，结束字符记住模式串的字典序, 即字典排序后的数组下标
        currentState.addEmit(index)
        self.l[index] = len(keyword)

    def buildDoubleArrayTrie(self, keySet):
        """构建双数组树"""
        self.progress = 0
        self.keySize = len(keySet)
        self.resize(65536 * 32)
            
        self.base[0] = 1
        self.nextCheckPos = 0

        root_node = self.rootState
        siblings  = self.fetch(root_node)

        if len(siblings) == 0:
            self.check = [-1] * len(self.check)
        else:
            self.insert(siblings)
    
    @abstractmethod
    def fetch(self, parnt, siblings):
        pass

    def insert(self, siblings):
        """
        插入节点
        -param list siblings 等待插入的兄弟节点
         当前siblings和双字典树有所不同，格式为[int, State]，不过，int 的值不变，还是父节点unioncode + 1        
        -return int 插入位置
        """
        
        # pos 来自当前兄弟节点中第一个节点的hash值或父节点最后
        begin = 0
        pos = max(siblings[0][0] + 1, (self.nextCheckPos + 1) if self.enableFastBuild else self.nextCheckPos) - 1
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
            begin = pos - siblings[0][0] 
            
            # 保证起始位置开始后面有足够的空间，为下面的循环做铺垫
            if self.allocSize <= (begin + siblings[len(siblings) - 1][0]):
                self.resize(begin + siblings[len(siblings) - 1][0] + 65535)
            
            if self.used[begin]:
                continue
            
            # 这种循环方式是为了实现 java 中的 continue outer；当内循环正常终止，才终止外循环，否则继续外循环
            # 这句话的意思，如果找到了满足目标空闲空间，就跳出整个循环
            for i in range(1, len(siblings)):
                if self.check[begin + siblings[i][0]] != 0:
                    break
            else:
                break

        """
        启发式空闲搜索法
        从位置nextCheckPos开始到pos之间，如果占用率在95%以上，下次插入节点时，直接从pos位置开始查找 
        """ 
        if 1.0 * nonzero_num / (pos - self.nextCheckPos + 1) >= 0.95:
            self.nextCheckPos = pos

        self.used[begin] = True

        if self.size <= begin + siblings[len(siblings) - 1][0] + 1:
            self.size = begin + siblings[len(siblings) - 1][0] + 1

        # 检查每个子节点 
        # 给 check 赋值，建立子节点 (check[begin + s.code]) 与父节点 (begin) 的多对一的关系
        # check的对应位置是：字符hash值 + 1 ，其赋的值是：上一个字符的位置(父节点)，初始等于1
        for i in range(len(siblings)):
            self.check[begin + siblings[i][0]] = begin
            self.char[begin + siblings[i][0]] = siblings[i].c
            #print(f'====insert-Middle=【begin={begin}】【({i}), char={siblings[i].c} code={siblings[i][0]}】【pos={pos}】')
        
        # 检查每个子节点, 若其没有孩子，就将它的base设置 -1，否则就调用 insert 建立关系 
        # 给 base 赋值，建立父节点 (base[begin + s.code]) 与子节点 (h) 的一对多的关系
        for i in range(len(siblings)):
            new_siblings = self.fetch(siblings[i])
            
            # 一个词的终止且不为其他词的前缀
            if len(new_siblings) == 0:
                self.base[begin + siblings[i][0]] = (-siblings[i][1].getLargestValueId() - 1) 
                self.progress += 1

            else:
               h = self.insert(new_siblings)
               self.base[begin + siblings[i][0]] = h
            
            # 让每个状态记住自己的双数组下标
            siblings[i][1].setIndex(begin + siblings[i][0])

        return begin

    def constructFailureStates(self):
        """建立failure表"""
        self.fail = [0] * (self.size + 1)
        self.output = np.zeors((self.size, 1))
        
        queue = deque()

        # 第一步，将深度为1的节点的failure设为根节点
        for depthOneState in self.rootState.getStates():
            depthOneState.setFailure(self.rootState, self.fail)
            queue.append(depthOneState)
            
            self.constructOutput(depthOneState)

        # 第二步，将深度 > 1 的节点建立failure表，这是一个广度优先遍历（bfs）
        while len(queue) > 0:
            currentState = queue.popleft()
            for transition in currentState.getTransitions():
                
                # 当前状态的下一个状态
                targetState = currentState.nextState(transition)
                queue.append(targetState)
                
                # 当前状态的失败转移状态，若失败转移状态没有子状态，则获取失败转移状态中的失败转移状态
                traceFailureState = currentState.getFailure()
                while traceFailureState and not traceFailureState.nextState(transition):
                    traceFailureState = traceFailureState.getFailure()
                
                newFailureState = traceFailureState.nextState(transition)
                targetState.setFailure(newFailureState, self.fail)
                targetState.addEmit(newFailureState.emit())

                self.constructOutput(targetState)
    
    def constructOutput(self, targetState):
        """建立output表"""
        emit = targetState.getEmit()
        if emit == None or len(emit) <= 0:
            return

        output = [0] * len(emit)
        for i, it in enumerate(emit):
            output[i] = it

        self.output[targetState.getIndex()] = output

    def resize(self, newSize):
        """拓展数组"""
        base2  = [0] * newSize
        check2 = [0] * newSize
        char2  = [0] * newSize
        used2  = [False] * newSize

        if self.allocSize > 0:
            base2   = self.base[:self.allocSize]
            check2  = self.check[:self.allocSize]
            char2   = self.char[:self.allocSize]
            used2   = self.used[:self.allocSize]
        
        self.base  = base2
        self.check = check2
        self.char  = char2
        self.used  = used2
        
        self.allocSize = newSize

    def loseWeight(self):
        """释放空闲内存"""
        nbase = [0] * (self.size + 65535)
        nbase = self.base[:self.size]

        self.base = nbase

        ncheck = [0] * (self.size + 65535)
        ncheck = self.check[:min(len(self.check), len(ncheck))]

        self.check = ncheck
    

class AhoCorasickDoubleArrayTrie(Builder):

    enableFastBuild = False

    def __init__(self, dictionary, enableFastBuild):
        self.enableFastBuild = enableFastBuild
        self.build(dictionary)
    
    def parseText(self, text):
        currentState, collectedEmits = 0, []
        for i in len(text):
            currentState  = self.getState(currentState, text[i])
            collecedEmits = self.storeEmits(i + 1, currentState)

        return collectedEmits

    def build(self, treeMap):
        Build().build(treeMap)

    def fetch(self, parent) -> list:
        """
        获取直接相连的子节点
        -param parent 父节点
        return 兄弟节点
        """
        siblings = []
        if parent.isAcceptable():
            # 此节点是parent的子节点，同时具备parent的输出
            fakeNode = State(-(parent.getDepth() + 1))
            fakeNode.addEmit(parent.getLargesValueId())

            siblings.add((0, fakeNode))
            
        # entry.getKey() 应该是字符才对，在这里进行运算操作，实际上是对该字符Unicode进行运算
        for entry in parent.getSuccess():
            siblings.add((entry.getKey() + 1, entry.getValue()))

        return siblings
    
    def getState(self, currentState, character):
        """状态转移，支持failure转移"""
        newCurrentState = self.transitionWithRoot(currentState, character)
        while newCurrentState == -1:
            currentState = self.fail[currentState]
            newCurrentState = self.transitionWithRoot(currentState, character)

        return newCurrentState

    def storeEmits(self, position, currentState):
        """保存输出"""
        hitArray = self.output[currentState]

        collectedEmits = []
        if hitArray:
            for hit in hitArray:
               collectedEmits.append((position - self.l[hit], position, self.v[hit]))

        return collectedEmits

    def transitionWithroot(self, _from, c):
        b = self.base[_from]
        p = b + c + 1

        if b != self.check[p]:
            if _from == 0:
                return 0

            return -1

        return p


