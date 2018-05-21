import numpy as np
import heapq
import math
import time

class Embedding:
    
    def __init__(self, path):
        self.m = self.normalize(np.load(path + '.npy'))
        self.dim = self.m.shape[1]
        self.wi, self.iw = self.readVocab(path + '.vocab')
        
    
    def zeros(self):
        return np.zeros(self.dim)
    
    def dimension(self):
        return self.dim
    
    def normalize(self, m):
        norm = np.sqrt(np.sum(m*m, axis=1))
        norm[norm==0] = 1
        return m / norm[:, np.newaxis]
    
    def readVocab(self, path):
        vocab = []
        with open(path) as f:
            for line in f:
                vocab.extend(line.strip().split())
        return dict([(w, i) for i, w in enumerate(vocab)]), vocab
    
    def __contains__(self, w):
        return w in self.wi
        
    def represent(self, w):
        return self.m[self.wi[w], :]
    
    def scores(self, vec):
        return np.dot(self.m, vec)

    # why +1 .../2?
    def pos_scores(self, vec):
        return (np.dot(self.m, vec)+1)/2

    def pos_scores2(self, vec):
        scores = np.dot(self.m, vec)
        scores[scores < 0.0] = 0.0
        return scores


    def top_scores(self, scores, n=10):
        if n <= 0:
            n = len(scores)
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])
    
    def closest(self, w, n=10):                        
        scores = np.dot(self.m, self.represent(w))               
        return self.top_scores(scores,n)

    def closest_with_time(self, w, n=10):        
        start = time.time()                
        scores = np.dot(self.m, self.represent(w))
        end = time.time()        
#        print "\nDeltatime: %f msec\n" % ((end-start)*1000)
        return self.top_scores(scores,n), end-start

    def closest_vec(self, wordvec, n=10):
        #scores = self.m.dot(self.represent(w))
        scores = np.dot(self.m, wordvec)
        return self.top_scores(scores, n)
#        if n <= 0:
#            n = len(scores)
#        return heapq.nlargest(n, zip(self.iw, scores))
    
    def closest_vec_filtered(self, wordvec, vocab, n=10):
        scores = np.dot(self.m, wordvec)
        if n <= 0:
            n = len(scores)
        scores_words = zip(self.iw, scores)
        for i in xrange(0,len(scores_words)):
            if not scores_words[i][1] in vocab: 
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])
      
    def closest_prefix(self, w, prefix, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in xrange(0,len(scores_words)):
            if not scores_words[i][1].startswith(prefix): 
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])
    
    def closest_filtered(self, w, vocab, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in xrange(0,len(scores_words)):
            if not scores_words[i][1] in vocab: 
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])
 
    def similarity(self, w1, w2):
        return self.represent(w1).dot(self.represent(w2))    

def norm_vec(vec):
    length = 1.0 * math.sqrt(sum(val ** 2 for val in vec))
    return [val/length for val in vec]

def score2string(score):
    return score[1] + "\t" + '{0:1.3f}'.format(score[0])


def closest_sym_scores(targets, subs, w, n):
    w_target_vec = targets.represent(w)
    w_sub_vec = subs.represent(w)
    w2subs = subs.closest_vec(w_target_vec,0)
    w2subs2w = []
    for entry in w2subs:
        score = (entry[0]+1)/2
        sub = entry[1]
        sub_target_vec = targets.represent(sub)       
        rev_score = (np.dot(sub_target_vec, w_sub_vec)+1)/2
        w2subs2w.append((math.sqrt(score * rev_score), sub))
    return heapq.nlargest(n, w2subs2w)
    
    
