from tree import *
import numpy as np

def count_fn(x):
    if x.isLeaf:
        if x.word in dict:
            dict[x.word] += 1
        else:
            dict[x.word] = 1

t = Tree()
t.loadTrees()
itr = t.getTree('train')
dict = {}
train_count=0
for x in itr:
    t.postOrderTraversal(x,count_fn)
    train_count += 1
print "Train count: ",train_count

data = open('glove.6B.300d.txt','r').readlines()
glove = {}
for line in data:
    line = line.strip().split()
    glove[line[0]] = np.asarray(line[1:],dtype=np.float32)

# in_glove = {}
# vocab = list(set(glove.keys()).intersection(set(dict.keys())))
# glove['<unk>'] = np.random.randn(300).astype(np.float32)
# vocab += ['<unk>']
# emb = np.asarray([glove[word] for word in vocab],dtype=np.float32)

vocab_count = sorted(list(dict.iteritems()),key=lambda tup:tup[1],reverse=True)
vocab = [x for x,y in vocab_count] + ['<unk>']
idx_to_word = {i:x for i,x in enumerate(vocab)}
word_to_idx = {y:x for x,y in idx_to_word.iteritems()}
emb = 0.2 * np.random.normal(scale=np.sqrt(1/300.),size=(len(vocab), 300))
count=0
print "Vocab size: ",len(vocab)
for word in glove.keys():
    if word in word_to_idx:
        count+=1
        idx = word_to_idx[word]
        emb[idx] = glove[word].copy()
print "No. of words found in GloVE: ",count

props = {}
props['idx_to_word'] = idx_to_word
props['word_to_idx'] = word_to_idx
props['vocab_count'] = len(vocab)
props['vocab'] = vocab
props['embeddings'] = emb
np.save('properties',props)
