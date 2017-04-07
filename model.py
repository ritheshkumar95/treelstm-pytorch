import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torch.autograd import Variable
from tree import *
import time

VOCAB_SIZE = 16582
EMB_DIM = 300
LSTM_DIM = 150
OUTPUT_DIM = 5
TRAIN_ITER = 0
BATCH_SIZE = 25

states=[]
labels=[]
T = Tree(use_vocab=True)
T.loadTrees()
props = np.load('properties.npy').tolist()

class Network:
    def __init__(self):
        self.model = {}
        self.model['word_emb'] = nn.Embedding(VOCAB_SIZE,EMB_DIM)
        self.model['linear_input'] = nn.Linear(EMB_DIM,LSTM_DIM*5,bias=True)
        self.model['linear_hidden'] = nn.Linear(LSTM_DIM*2,LSTM_DIM*5)
        self.model['linear_output'] = nn.Linear(LSTM_DIM,OUTPUT_DIM)

    def move_to_gpu(self):
        for item in self.model.values():
            item.cuda()

    def getParameters(self):
        params = []
        for layer in self.model:
            for param in self.model[layer].parameters():
                params += [param]
        return params

    def initParameters(self):
        for layer in self.model:
            if 'emb' in layer:
                continue
            for param in self.model[layer].parameters():
                if param.dim()==1:
                    nn.init.constant(param,0.)
                else:
                    nn.init.xavier_uniform(param)

    def zero_grad(self):
        for layer in self.model:
            model[layer].zero_grad()

net = Network()
net.move_to_gpu()
net.initParameters()
model = net.model
params = net.getParameters()
model['optimizer'] = torch.optim.Adagrad(params,lr=0.05, weight_decay=1e-4)
emb_param = list(model['word_emb'].parameters())[0]
emb_param.data = emb_param.data.copy_(torch.from_numpy(props['embeddings']).cuda())
print "Embeddings copied!"

def state_update(state_left,state_right,input=None):
    h_left,c_left = state_left.split(LSTM_DIM,1)
    h_right,c_right = state_right.split(LSTM_DIM,1)

    if input:
        gates = model['linear_input'](input)
    else:
        gates = model['linear_hidden'](torch.cat([h_left,h_right],1))

    i_t,f_t_left,f_t_right,o_t = nn.Sigmoid()(gates[:,:4*LSTM_DIM]).split(LSTM_DIM,1)
    u_t = nn.Tanh()(gates[:,4*LSTM_DIM:])

    c_t = i_t*u_t + f_t_left*c_left + f_t_right*c_right
    h_t = o_t * nn.Tanh()(c_t)
    return torch.cat((h_t,c_t),1)


def inference(node):
    global states,labels
    if node.isLeaf:
        current_input = model['word_emb'](Variable(torch.cuda.LongTensor([props['word_to_idx'][node.word]]),requires_grad=False))
        state_left = Variable(torch.cuda.FloatTensor(1,2*LSTM_DIM).fill_(0.),requires_grad=False)
        state_right = Variable(torch.cuda.FloatTensor(1,2*LSTM_DIM).fill_(0.),requires_grad=False)
    else:
        state_left = node.left.state
        state_right = node.right.state
        current_input = None
    node.state = state_update(state_left,state_right,current_input)
    states.append(node.state)
    labels.append(node.label)

def train_step(node,train=True):
    drop_prob = 0.5 if train else 0.0
    global states,labels,TRAIN_ITER
    states = []
    labels = []
    T.postOrderTraversal(node,inference)
    h_ts = nn.Dropout(drop_prob)(torch.cat(states,0).split(LSTM_DIM,1)[0])
    out = model['linear_output'](h_ts)
    target = Variable(torch.cuda.LongTensor(labels),requires_grad=False)

    # weights = torch.cuda.FloatTensor([ 0.46479052,  0.1115243 ,  0.01743588,  0.08671308,  0.31953622])
    loss = nn.CrossEntropyLoss()(out,target)

    logits = nn.Softmax()(out)
    _,sents = torch.max(logits,1)
    sents = np.asarray(sents.data.tolist()).reshape((-1))
    acc = sents[-1]==labels[-1]
    if train:
        TRAIN_ITER += 1
        loss.backward()
        if TRAIN_ITER%BATCH_SIZE==0:
            model['optimizer'].step()
            model['optimizer'].zero_grad()
    return loss,acc,sents

def score(set='valid'):
    itr = T.getTree(set)
    iter = 0
    costs = []
    accuracy = []
    times = []
    for node in itr:
        iter += 1
        start = time.time()
        loss,acc,_ = train_step(node,train=False)
        costs += loss.data.tolist()
        accuracy += [acc]
        times += [time.time()-start]

    print "Validation Completed!"
    print "\t %s cost    : %f" %(set,np.mean(costs))
    print "\t %s accuracy: %f" %(set,np.mean(accuracy))
    print "\t Total time     : ", np.sum(times)
    print ""

words = []
def predict(set='test'):
    global words
    words = []
    test_labels = []
    itr = T.getTree(set)
    count = np.random.randint(100)
    iter=0
    for node in itr:
        iter += 1
        if iter==count:
            break
    loss,acc,predicted_labels = train_step(node,train=False)

    def print_fn(x):
        global words
        if x.isLeaf:
            words += [x.word]
    T.postOrderTraversal(node,print_fn)
    print "Sentence   :      ",words
    print "True labels:      ",labels
    print "Predicted labels: ",predicted_labels.tolist()

for i in xrange(50):
    itr = T.getTree('train')
    iter = 0
    TRAIN_ITER=0
    costs = []
    accuracy = []
    times = []
    for node in itr:
        iter += 1
        start = time.time()
        loss,acc,_ = train_step(node)
        costs += loss.data.tolist()
        accuracy += [acc]
        times += [time.time()-start]

        if iter%1024==0:
            print "Iter %d: (Epoch %d)"%(iter,i)
            print "\t Train cost    : ", np.mean(costs)
            print "\t Train accuracy: ", np.mean(accuracy)
            print "\t Mean time     : ", np.mean(times)
            print ""

    score('valid')
