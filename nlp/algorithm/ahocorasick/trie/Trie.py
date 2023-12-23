from .State import State 
from .TrieConfig import TrieConfig
from collections import deque

class Trie():
    
    trieConfig = None
    rootState  = None

    failureStatesConstructed = False

    def __init__(self, keywords = [], trieConfig = None):
        if not trieConfig:
            self.trieConfig = TrieConfig()

        self.rootState  = State()
        
        if len(keywords) > 0:
            self.addAllKeyword(keywords)

        #print('---',self.rootState.toString())

    def addKeyword(self, keyword):
        if not keyword or len(keyword) <= 0:
            return
        
        # 构建 goto 表，前缀树
        currentState = self.rootState
        for character in keyword:
            currentState = currentState.addState(character)
                    
        # 构建 output 表，保存结束字符的模式串
        currentState.addEmit(keyword)

    def addAllKeyword(self, keywords):
        for keyword in keywords:
            self.addKeyword(keyword)

    def parseText(self, text):
        self.checkForConstructedFailureStates()

        position = 0
        currentState = self.rootState
        
        collectedEmits = []
        for i in range(len(text)):
            currentState = self.getState(currentState, text[i])
            collectedEmits.extend(self.storeEmits(position, currentState))
            
            position += 1

        # 不允许重叠
        #if not self.trieConfig.isAllowOverlaps():
        #    intervalTree = collectedEmits
        #    intervalTree.remove
        
        # 只保留最长词
        if self.trieConfig.remainLongest:
            collectedEmits = self.remainLongest(collectedEmits)

        return collectedEmits

    
    def remainLongest(self, collectedEmits):
        """只保留最长词"""
        if len(collecteEmits) < 2:
            return

        emitMapStart = {}
        for emit in collectedEmits:
            pre = emitMapStart.get(emit[0])
            if not pre or pre.size() < emit[1] - emit[0] + 1:
                emitMapStart[emit[0]] = emit

        if len(emitMapStart) < 2:
            collectedEmits = emitMapStart.values()
            return collectedEmits
    
        emitMapEnd = {}
        for emit in emitMapStart.values():
            pre = emitMapEnd.get(emit[1])
            if not pre or pre.size() < emit[1] - emit[0] + 1:
                emitMapEnd[emit[1]] = emit

        collectedEmits = emitMapEnd.values()
        return collectedEmits
    
    def getState(self, currentState, character):
        """
        跳转到下一个状态
        @param currentState 当前状态
        @param character    接受字符
        @return 跳转结果
        """
        newCurrentState = currentState.nextState(character)
        while not newCurrentState:
            currentState = currentState.getFailure()
            newCurrentState = currentState.nextState(character)

        return newCurrentState
    
    def checkForConstructedFailureStates(self) -> bool:
        """检查是否建立了failure表"""
        if not self.failureStatesConstructed:
            self.constructFailureStates()

    def constructFailureStates(self):
        """建立failure表"""
        queue = deque()
        
        # 第一步，将深度为1的节点的failure设为根节点
        for depthOneState in self.rootState.getStates():
            depthOneState.setFailure(self.rootState)
            queue.append(depthOneState)

        self.failureStatesConstructed = True

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
                targetState.setFailure(newFailureState)
                targetState.addEmit(newFailureState.emit())
    
    def storeEmits(self, position, currentState) -> list:
        """
        保存匹配结果
        @param position 当前位置，也就是匹配到的模式串的结束位置 +1
        @param currentStatue 当前状态
        """
        emits = currentState.emit()
        if not emits or len(emits) <= 0:
            return []
        
        collectedEmit = []
        for emit in emits:
            collectedEmit.append((position - len(emit) + 1, position, emit))

        return collectedEmit


            

