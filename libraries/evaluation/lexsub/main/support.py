import numpy as np
import re
from nltk.corpus import wordnet as wn
from pos import to_wordnet_pos
from nltk.stem.wordnet import WordNetLemmatizer


def read_vectors(file, header=False):
    dict = {}
    with open(file, 'r') as f:
            for i, sentence in enumerate(f):
                if header and i == 0:
                    continue
                parts = sentence.strip().split(" ")
                word = parts[0]
                vec = np.array(parts[1:], dtype="float32")
                # normalize
                vec = vec/np.sum(vec**2)**0.5
                dict[word] = vec
    return dict


def conll_skip_sentence(conll, eos_symbol="<eol>"):
    while True:
        line = re.split(r'\t', conll.readline())
        word = line[1]
        if word == eos_symbol:
            break
    conll.readline()



def read_candidates(candidates_file, allowed_words):
    target2candidates = {}
    # finally.r::eventually;ultimately
    print "--- reading candidates ---"
    with open(candidates_file, 'r') as f:
        for i, line in enumerate(f):
            # if (i+1) % 1 == 0:
            #     print 'read %d lines' %(i+1)
            segments = line.strip().split('::')
            target = segments[0]
            word, pos = target.split('.')
            # assuming that candidates are unique initially
            candidates = [str(c) for c in segments[1].split(';') if c.find(" ") == -1]  # forbid composite words
            candidates = filter_candidate(pos, candidates, allowed_words)
            target2candidates[target] = candidates
    print '--- done ---'
    return target2candidates


# performs a 3 step filtering by matching with vocabulary of allowed_words
# note that candidates is a dictionary old_cand -> [words in vocab], where new_cands are candidates
# we find matching our allowed_words
def filter_candidate(pos, candidates, allowed_words):
    # this fixes the problem with non_wordnet pos tags
    if pos in to_wordnet_pos:
        pos = to_wordnet_pos[pos]
    new_candidates = {}
    for word in allowed_words.keys():
        if not is_ascii(word):
            continue
        lemma = WordNetLemmatizer().lemmatize(word, pos)
        # we try 3 things
        tries = [lemma, lemma.title, word, word.title()]
        # tries = [lemma]
        for w in tries:
            if w in candidates:
                if w not in new_candidates:
                    new_candidates[w] = []
                new_candidates[w].append(word)
    return new_candidates

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def cosine_sim(x, y):
    return float(np.sum(x*y))/float(np.sqrt(np.sum(x**2)*np.sum(y**2)))

# some wierd form of dot pos dot product
# with normalized vectors
def pos_cosine_sim_normed(x, y):
        return (np.dot(x, y)+1)/2

def wf2ws(weight):
        return '{0:1.5f}'.format(weight)



def morphify(word, pos):
    """ morph a word """
    synsets = wn.synsets(word, pos=pos)

    # Word not found
    if not synsets:
        return []

    # Get all  lemmas of the word
    lemmas = [l for s in synsets for l in s.lemmas() if s.name().split('.')[1] == pos]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the targeted pos
    related_lemmas = [l for drf in derivationally_related_forms \
                           for l in drf[1] if l.synset().name().split('.')[1] == pos]

    # Extract the words from the lemmas
    words = [l.name() for l in related_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result


def flatten(list):
    return [item for sublist in list for item in sublist]

# returns (candidate, best_score)
def get_best_scores_for_candidates(candidates, scores):
    new_scores = {}
    for cand, words in candidates.items():
        # now choose the best one
        new_scores[cand] = max([scores[w] for w in words])
    return new_scores

