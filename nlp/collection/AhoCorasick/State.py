"""
一个状态有以下几个功能：
1，success：成功转移到另外一个状态
2，failure：不可顺着字符串跳转的话，则跳转到另外一个浅一点的节点
3，emits：命中一个模式串
"""

class State():
    
    # 模式串的长度，也是这个状态的深度
    depth = 0
    
    # fail 函数，如果没有匹配到，则跳转到此状态。
    failure = None
    
    # output 表，只要这个状态可达，则记录模式串
    emits = None
    
    # goto 表，也称转移函数。根据字符串的下一个字符转移到下一个状态
    success = None
    
    # 在双数组中的对应下标
    index = 0

    def __init__(self, depth = 0):
        self.depth = depth
        self.success = {}
    
    def getDepth(self):
        return self.depth

    def addEmit(self, keyword):
        """添加一个匹配到的模式串（这个状态对应着这个模式串)"""
        
        if self.emits == None:
            self.emits = []

        if isinstance(keyword, list):
            for emit in keyword:
                self.addEmit(emit)
        else: 
            self.emits.append(keyword)
    
    def getEmit(self):
        """获取这个节点代表的模式串（们）"""
        return [] if self.emits == None else self.emits
    
    def getLargestValueId(self) -> int:
        """获取最大的值"""
        if self.emits == None or len(self.emits) <= 0:
            return None
        
        return max(self.emits)

    def isAcceptable(self):
        """是否是终止状态"""
        return self.depth > 0 and self.emits != None

    def nextState(self, character, ignoreRootState = False):
        """
        转移到下一个状态
        @param character 希望按此字符转移
        @param ignoreRootState 是否忽略根节点，如果是根节点自己调用则应该是true，否则为false
        @return 转移结果
        """
        nextState = self.success.get(character)
        if not ignoreRootState and not nextState and self.depth == 0:
            nextState = self

        return nextState
    
    def nextStateIgnoreRootState(self, character):
        """按照character转移，任何节点转移失败会返回null"""
        return self.nextState(character, True)

    def addState(self, character):
        nextState = self.nextStateIgnoreRootState(character)
        if not nextState:
            nextState = State(self.depth + 1)
            self.success[character] = nextState

        return nextState

    def getStates(self) -> list:
        return self.success.values()

    def getTransitions(self) -> list:
        return self.success.keys()
    
    def getFailure(self):
        """获取failure状态"""
        return self.failure

    def setFailure(self, failState):
        """设置failure状态"""
        self.failure = failState
        
        # 传递回去, 这是和 algorithm.State 不同的地方
        #fail[self.index] = self.failure.index
    
    def setIndex(self, index):
        self.index = index

    def getIndex(self):
        return self.index

    def toString(self):
        sb = 'State{' \
                f'depth={self.depth}' \
                f',emits={self.emits}' \
                f',success={self.success.keys()}' \
                f',failureId={"-1" if self.failure == null else self.failure.index}' \
                f',failure={self.failure}' \
                '}'

        return sb

    def getSuccess(self):
        """获取goto表"""
        return self.success

