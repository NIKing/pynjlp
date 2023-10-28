from .BaseNode import BaseNode, Status
from .util.ArrayTool import binarySearch

class Node(BaseNode):
    def __init__(self, char = '', status = '', value = None) -> None:
        super().__init__(char, status, value)

        self.c = char
        self.value  = value
        self.status = status

    def getChild(self, c):
        if not self.child:
            return None

        index = binarySearch(self.child, c)
        #print(f'!!!!!c_c={self.c}，child_len={len(self.child)}, t_c={c}, node_child={self.child},  index={index}')

        if index < 0:
            return None
        
        return self.child[index]

    def addChild(self, node):
        add = False

        child = self.child
        if child == None:
            child = []

        index = binarySearch(child, node)
        #print(f'+++++添加{node.getChar()}, 二分查询={index}')

        if index >= 0:
            target = child[index]

            if node.status == Status.UNDEFINED and target.status != Status.NOT_WORD:
                target.status = Status.NOT_WORD
                target.value  = None

            elif node.status == Status.NOT_WORD and target.status == Status.WORD_END:
                target.status = Status.WORD_MIDDLE

            elif node.status == Status.WORD_END and target.status != Status.WORD_END:
                target.status = Status.WORD_MIDDLE

                if not target.getValue():
                    add = True

                target.setValue(node.getValue())

        else:
            
            newChild = [Node()] * (len(child) + 1)
            
            # 这是新节点的位置，可能是在最后，可能是child中间
            insert = -(index + 1)
            
            # 位置替换
            #newChild = child[:insert] + [Node()] * (len(child) - insert) + [Node()]
            newChild[:insert]   =  child[:insert]
            newChild[insert+1:] =  child[insert:]
            newChild[insert]    =  node
           
            add = True
            self.child = newChild

        return add



