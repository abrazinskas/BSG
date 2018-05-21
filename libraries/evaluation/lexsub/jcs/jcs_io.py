import math
import heapq

STOPWORD_TOP_THRESHOLD = 256
SUBVEC_DIR_SUFFIX = ".DIR"
VOCAB_TOTAL = "<TOTAL>"

def wf2ws(weight):
        return '{0:1.5f}'.format(weight)
    

def vec_to_str(subvec, max_n):
    
    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [' '.join([word, wf2ws(weight)]) for word, weight in sub_list_sorted]
    return '\t'.join(sub_strs)

def vec_to_str_generated(subvec, max_n):
    
    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [word for word, weight in sub_list_sorted]
    return ';'.join(sub_strs)

def count_file_lines(filename):
    f = open(filename, 'r')
    lines_num = sum(1 for line in f)
    f.close()
    return lines_num

def to_rank_weights(subvec):
    subvec_len = len(subvec)
    for i in xrange(0, subvec_len):
        subvec[i] = (subvec[i][0], 1.0-float(i)/subvec_len)
                        
def get_pmi_weights(subvec, w2counts, sum_counts, offset, threshold, normalize=False):
    subvec_pmi = []
    norm = 0
    for word, prob in subvec:
        if prob != 0.0:
            pmi = math.log(prob * sum_counts / w2counts[word])-offset
            if pmi>threshold:
                subvec_pmi.append((word, pmi))
                norm += pmi**2
            
    if normalize:
        norm = norm**0.5
        for i in xrange(0,len(subvec_pmi)):
            subvec_pmi[i] = (subvec_pmi[i][0], subvec_pmi[i][1] / norm)       
            
    return subvec_pmi

        
    
def extract_word_weight(pair):
    tokens = pair.split(' ')
    return tokens[0], float(tokens[1])


def load_classes(path):
    w2c = {}
    max_class_id = 0
    with open(path) as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            class_id = int(tokens[1])  
            w2c[word] = class_id
            max_class_id = max(max_class_id, class_id) 
    return w2c, max_class_id+1

def load_vocabulary_w2i(path):
    with open(path) as f:
        vocab = [line.split('\t')[0].strip() for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab

def load_vocabulary_counts(path, factor=1.0):
    stop_words = set()
    counts = {}
    sum = 0
    with open(path) as f:
        i = 0
        for line in f:
            if len(line) > 0:
                tokens = line.split('\t')
                # tokens = line.split(' ')
                word = tokens[0].strip()
                count = int(tokens[1].strip())
                if (factor != 1.0):           
                    factored_count = int(count**factor)
                else:
                    factored_count = count
                counts[word] = factored_count
                sum += factored_count
                i += 1
                # What is this?!
                if (i <= STOPWORD_TOP_THRESHOLD):
                    stop_words.add(word)
    total_size = sum #counts[VOCAB_TOTAL]
    return counts, total_size, stop_words

def load_target_counts(path):
    counts = {}
    with open(path) as f:
        for line in f:
            if len(line) > 0:
                tokens = line.split('\t') 
                word = tokens[0].strip() 
                count = int(tokens[1].strip())
                counts[word] = count
    return counts