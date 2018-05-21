'''
Convert word embeddings from text files to numpy-friendly format
'''

import numpy as np
import sys


def readVectors(path, header=False):
    vectors = {}
    with open(path) as input_f:
        for i, line in enumerate(input_f.readlines()):
            if header and i==0:
                continue
            if line == "":
                continue
            tokens = line.strip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return vectors

inpath = sys.argv[1]
outpath = sys.argv[2]
header = True if sys.argv[3]=="True" else False

matrix = readVectors(inpath, header=header)


print "done reading vectors"
vocab = list(matrix.keys())
vocab.sort()
with open(outpath+'.vocab', 'w') as output_f:
    for word in vocab:
        print >>output_f, word,

new_matrix = np.zeros(shape=(len(vocab), len(matrix[vocab[0]])), dtype=np.float32)
for i, word in enumerate(vocab):
    if i%1000 == 0:
        print(i)
    new_matrix[i, :] = matrix[word]

print new_matrix.shape

np.save(outpath+'.npy', new_matrix)