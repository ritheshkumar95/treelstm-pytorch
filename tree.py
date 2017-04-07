import numpy as np
class Node:
    def __init__(self,lab,lChild=None,rChild=None):
        self.label = lab
        self.left = lChild
        self.right = rChild
        self.parent = None
        self.isLeaf = False
        self.word = None
        self.children = []

class Tree:
    def __init__(self,use_vocab=False):
        self.open = '('
        self.close = ')'
        self.root = None
        self.data = None
        self.use_vocab = use_vocab
        if use_vocab:
            self.vocab = np.load('properties.npy').tolist()['vocab']

    def parse(self,tokens,parent=None):
        assert tokens[0]  == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1

        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1]))  # zero index labels
        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower()
            if self.use_vocab:
                if node.word not in self.vocab:
                    node.word = '<unk>'
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        # print "left: ",tokens[2:split]
        node.right = self.parse(tokens[split:-1], parent=node)
        # print "right: ",tokens[split:-1]

        if parent is None:
            self.root = node

        return node

    def postOrderTraversal(self,node,applyFn,args=[]):
        if node is None:
            return

        if node.left:
            self.postOrderTraversal(node.left,applyFn,args)

        if node.right:
            self.postOrderTraversal(node.right,applyFn,args)

        applyFn(node,*args)

    def loadTrees(self):
        f = open("trees/train.txt","r")
        self.data = {}
        self.data['train'] = f.readlines()

        f = open("trees/dev.txt","r")
        self.data['valid'] = f.readlines()

        f = open("trees/test.txt","r")
        self.data['test'] = f.readlines()


    def getTree(self,set='train'):
        data = self.data[set]
        for i in xrange(len(data)):
            tmp = self.parse(data[i].strip().replace(' ',''))
            yield tmp
