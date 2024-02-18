from nlp.collection.trie.bintrie.BaseNode import BaseNode, Status
from nlp.collection.trie.bintrie.Node import Node
from nlp.collection.trie.bintrie.HashCode import hash_code

class BinTrie(BaseNode):

    def __init__(self, map_data = None):
        super().__init__(self)

        self.child = [Node()] * (65535 + 1)
        self.size = 0
        self.status = Status.NOT_WORD
        
        if map_data == None:
            return None
        
        keys = list(map_data.keys())
        for i, key in enumerate(keys):
            value = map_data[key]
            self.put(key, value)

    def put(self, key, value):
        """插入一个词"""
        if not key:
            return None
        
        #print('----------start----------')
        #print(f'key={key}, len={len(key)}') 

        branch = self
        for char in key[:len(key) - 1]:
            #print(branch.getChar(), branch)
            #print('search--', char)

            branch.addChild(Node(char, Status.NOT_WORD, None))
            branch = branch.getChild(char)
        
        # 最后一个字符，增加属性
        branch.addChild(Node(key[len(key) - 1], Status.WORD_END, value))
        self.size += 1
        #print('-----------end-----------')
    
    def get(self, key): 
        branch = self
        for c in key:
            if not branch:
                return None

            branch = branch.getChild(c)
        
        if not branch:
            return None
        
        # 这句可以保证只有成词的节点被返回
        if branch.status != Status.WORD_END and branch.status != Status.WORD_MIDDLE:
            return None

        return branch.getValue()


    def getChild(self, c):
        return self.child[hash_code(c)]

    def addChild(self, node):
        add = False

        c = node.getChar()
        target = self.getChild(c)
        
        if not target.getChar():
            self.child[hash_code(c)] = node
            return True
        
        # 找到了目标节点，根据节点状态更新节点信息
        if node.status == Status.UNDEFINED and target.status != Status.NOT_WORD:
            target.status = Status.NOT_WORD
            add = True

        elif node.status == Status.NOT_WORD and target.status == Status.WORD_END:
            target.status = Status.WORD_MIDDLE

        elif node.status == Status.WORD_END:
            if target.status == Status.NOT_WORD:
                target.status = Status.WORD_MIDDLE

            if not target.getValue():
                add = True

            target.setValue(node.getValue())

        return add

    def getChar(self):
        return 0 # 根节点没有char
    
    def getSize(self):
        return self.size
    
    def entrySet(self):
        """获取键值对集合"""
        entrySet = {}
        for node in self.child:
            if not node or not node.getChar():
                continue
            
            node.walk([], entrySet)

        return entrySet 

    def parseText(self, text):
        """匹配文本, 根据字典返回最短匹配，比如有'工'和'工信部', 返回'工'"""
        begin, text_length, word_list = 0, len(text), []
        for i, c in enumerate(text):

            state = self.transition(c)
            if state:
                
                value = state.getValue()
                if value:
                    word_list.append(text[begin:i+1])
                
                # 最后的字符
                if i == text_length - 1:
                    begin += 1
            else:

                begin += 1

        return word_list

    def parseLongestText(self, text):
        """匹配长文本，根据字典返回最长匹配，比如'工'和'工信部'，返回'工信部'""" 
        i, text_len, word_list = 0, len(text), []
        
        while i < text_len:
            state = self.transition(text[i])
            if state:
                to  = i + 1
                end = to
                value = state.getValue()

                for j, char in enumerate(text[to:]):
                    state = state.transition(char)
                    if state == None:
                        break
                    
                    #print(f'--i={i},,j={j}, value={state.getChar()}')
                    #print(state.getValue())
                    value = state.getValue()
                    if value:
                        end = to + j + 1

                if value:
                    word_list.append(text[i:end])
                    i += end - 1
                    continue
            
            i += 1

        return word_list







