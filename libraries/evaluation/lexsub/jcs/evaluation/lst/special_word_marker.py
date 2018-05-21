
import sys
import re

RARE_WORD_TOKEN = '<RW>'
NUMERIC_TOKEN = '<NUM>'
NAME_TOKEN = '<NAME>'
MAX_COUNT_FOR_NAME = 10000

# very crude implementation
num_re = re.compile('^[\+\/\:\-,\.\d]*\d[\+\/\:\-,\.\d]*$')
def is_numeric(word_str):
    return num_re.match(word_str) != None

def is_name(word, word_lower, vocab, begin_sentence):
    isname = False
    if not begin_sentence:
        if word[:1].isupper():
            if word_lower not in vocab:
                isname = True
            else:
                count = vocab[word_lower]
                if count < MAX_COUNT_FOR_NAME:
                    isname = True
    return isname
    
def load_vocabulary(path):
    vocab = {}
    with open(path, 'r') as f:
        for line in f:
            if len(line) > 0:
                word = line.split('\t')[0].strip()
                count = int(line.split('\t')[1])
                vocab[word] = count                
    return vocab

            
def mark_special_words(words, start_ind, vocab):
    for i in xrange(start_ind, len(words)):
        if is_numeric(words[i]):
            words[i] = NUMERIC_TOKEN
        elif is_name(words[i], words[i].lower(), vocab, i==start_ind):
            words[i] = NAME_TOKEN
        else:
            words[i] = words[i].lower()
            
            

if __name__ == '__main__':
 
    if (len(sys.argv) < 2):
        print >> sys.stderr, "Usage: %s <vocab-file> <text >output"
        sys.exit(1)
        
    vocab = load_vocabulary(sys.argv[1])
    
    for line in sys.stdin:
        try:     
            segments = line.split('\t')
            words = segments[3].split()
            mark_special_words(words, 0, vocab)
            print '\t'.join(segments[:3]) + '\t' + ' '.join(words)
        except Exception as e:
            print >> sys.stderr, e
            sys.stderr.write("Can't parse line: %s" % line)
