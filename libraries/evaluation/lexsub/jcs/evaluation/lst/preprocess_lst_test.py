'''
'''

# &#8211;    -
# &#8212;    -
# &#8220;    "
# &#8221;    "
# &#8216;    '
# &#8217;    '

# &#8211 ;    -
# &#8212 ;    -
# &#8220 ;    "
# &#8221 ;    "
# &#8216 ;    '
# &#8217 ;    '

# &amp    &

import sys
import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag


first_quoted_re = re.compile('.*"(.*)".*')
context_re = re.compile('.*<context>(.*)</context>.*')
head_re = re.compile('.*<head>(.*)</head>.*')

target_prefix = '<lexelt item='
instance_prefix = '<instance id'
context_prefix = '<context>'

to_wordnet_pos = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV}
from_wordnet_pos = {wordnet.NOUN:'N',wordnet.ADJ:'J',wordnet.VERB:'V',wordnet.ADV:'R'}


def html_to_text(text):
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    
    text = text.replace('&#8211;', " - ")
    text = text.replace('&#8212;', " - ")
    text = text.replace('&#8220;', ' " ')
    text = text.replace('&#8221;', ' " ')
    text = text.replace('&#8216;', " '")
    text = text.replace('&#8217;', " '")
    text = text.replace('&#150;', " ")
    
    text = text.replace('&#8211 ;', " - ")
    text = text.replace('&#8212 ;', " - ")
    text = text.replace('&#8220 ;', ' " ')
    text = text.replace('&#8221 ;', ' " ')
    text = text.replace('&#8216 ;', " '")
    text = text.replace('&#8217 ;', " '")
    
    
    
    text = text.replace('&amp;', "&")
    
    return text
    
def to_lowercase(words):
    return [word.lower() for word in words]

def lemmatize(pairs):
    triples = []
    for pair in pairs:
        word = pair[0]
        pos = pair[1]
        wordnet_pos = wordnet.NOUN
        if (len(pos)>=2):
            pos_prefix = pos[:2]
            if (pos_prefix in to_wordnet_pos):
                wordnet_pos = to_wordnet_pos[pos_prefix]
        lemma = WordNetLemmatizer().lemmatize(word, wordnet_pos).lower();
        triples.append([word, wordnet_pos, lemma])
    return triples

def parse_context(context):
    target = head_re.match(context).group(1)
    tokens = context.split()
    target_ind = tokens.index('<head>'+target+'</head>')
    tokens[target_ind] = target
    
    return tokens, target_ind
    

def add_target(targets, target_with_pos, actual_target):
    wn_pos = target_with_pos.split('.')[-1]
    pos = from_wordnet_pos[wn_pos]
    targets.add(actual_target + "." + pos)

def is_atomic_mwe(mwe, verb_lemma, complement_lemma, synsets):
    mwe_count = 0
    for synset in synsets:
        gloss_lemmas = set([WordNetLemmatizer().lemmatize(word) for word in synset.definition.split()])
        if verb_lemma in gloss_lemmas or complement_lemma in gloss_lemmas:
            return False
        for syn_lemma in synset.lemmas:
            if syn_lemma.name != mwe: 
                tokens = syn_lemma.name.split('_')
                for token in tokens:
                    if token == verb_lemma:
                        return False
                if len(tokens) == 2 and tokens[1] == complement_lemma:
                    return False
        else:
            mwe_count += syn_lemma.count()
    return True   
                

def detect_mwe(text_tokens, target_ind, wordnet_pos):
    if (target_ind < len(text_tokens)-1):
        verb_lemma = WordNetLemmatizer().lemmatize(text_tokens[target_ind], wordnet_pos)
        complement_lemma = WordNetLemmatizer().lemmatize(text_tokens[target_ind+1])
        mwe = '_'.join([verb_lemma, complement_lemma])
        synsets = wordnet.synsets(mwe, wordnet.VERB) 
        if len(synsets) > 0:
            if (target_ind+1 < len(text_tokens)-1):
                mwe_right = '_'.join([WordNetLemmatizer().lemmatize(text_tokens[target_ind+1]), WordNetLemmatizer().lemmatize(text_tokens[target_ind+2])])
                if len(wordnet.synsets(mwe_right)) > 0:
                    return
            if is_atomic_mwe(mwe, verb_lemma, complement_lemma, synsets) == True:
                mwe = '='.join([text_tokens[target_ind], text_tokens[target_ind+1]])
                text_tokens[target_ind] = mwe
                del text_tokens[target_ind+1]
    
    
if __name__ == '__main__':
    
    if (len(sys.argv) > 1):
        input = open(sys.argv[1], 'r')
        output = open(sys.argv[2], 'w')
        detect_mwe_flag = False
        if (len(sys.argv) > 4):
            if sys.argv[4] == 'mwe':
                detect_mwe_flag = True        
    else:
        input = sys.stdin
        output = sys.stdout
        
    targets = set()
        
target = None
for line in input:
    
    line = line.strip()
    if line.startswith(target_prefix):
        target = first_quoted_re.match(line).group(1)        
    if line.startswith(instance_prefix):
        instance_id = first_quoted_re.match(line).group(1)
        continue
    if line.startswith(context_prefix):
        context = context_re.match(line).group(1)
        context = html_to_text(context)
        text_tokens, target_ind = parse_context(context)
        wn_pos = target.split('.')[-1]
        if wn_pos == wordnet.VERB and detect_mwe_flag:
            detect_mwe(text_tokens, target_ind, wordnet.VERB)
        add_target(targets, target, text_tokens[target_ind])
        text = ' '.join(text_tokens)
        output_line = '\t'.join([target, instance_id, str(target_ind), text])
        print >> output, output_line
        continue
    
if (len(sys.argv) > 0):
        input.close()
        output.close()
        
if (len(sys.argv) > 3):
    target_file = open(sys.argv[3], 'w')
    for target in targets:
        target_file.write(target + "\n")
    target_file.close()
            