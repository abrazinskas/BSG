'''
Instance in the Lexical Substitution Task dataset

'''


from pos import from_lst_pos
import re

CONTEXT_TEXT_BEGIN_INDEX = 3
TARGET_INDEX = 2

def encode_utf8(str):
    new_str = ""
    for c in str:
        try:
            new_c = c.__encode('utf-8')
            new_str+=new_c
        except UnicodeError:
            print "can't encode in utf-8"

    return new_str



class ContextInstance(object):

    def __init__(self, line):
        '''
        Constructor
        '''
        self.line = line
        tokens1 = line.split("\t")
        self.target_ind = int(tokens1[TARGET_INDEX])
        self.words = [w for w in tokens1[3].split()]
        self.target = self.words[self.target_ind]

        self.full_target_key = tokens1[0]
        self.pos = self.full_target_key.split('.')[-1]  # pos is last, but target_key contains the first pos
        self.target_key = '.'.join(self.full_target_key.split('.')[:2])  # remove suffix in cases of bar.n.v
        self.target_lemma = self.full_target_key.split('.')[0]       
        self.target_id = tokens1[1]
        if self.pos in from_lst_pos:
            self.pos = from_lst_pos[self.pos]

    # for non-parsed test data
    # TODO: modified version
    def get_neighbors(self, half_window_size):
        tokens = self.line.split()[3:]
        
        if half_window_size > 0:
            start_pos = max(self.target_ind-half_window_size, 0)
            end_pos = min(self.target_ind+half_window_size+1, len(tokens))
        else:
            start_pos = 0
            end_pos = len(tokens)

        left_neighbors = tokens[start_pos:self.target_ind]
        right_neighbors = tokens[self.target_ind+1:end_pos]
            
        neighbors = left_neighbors + right_neighbors
        return left_neighbors, right_neighbors

    # for parsed test data
    # returns all words that share a dependency label
    def get_dep_context(self, conll, target):
        id_to_word = []  # this one contains information for us to find out who point to the target
        ind_to_context_inds = {}  # it's like where our target points {word_ind =>[word_ind,...]}
        ind_to_prep_dep_inds = {}
        target_ind = None
        while True:
            line = conll.readline()
            parts = re.split(r"\t", line)
            ind, word, dep_word_ind, dep_type = int(parts[0]), parts[1], int(parts[6]), parts[7]
            if word == "<eol>":
                conll.readline()  # to just to the next sentence
                break
            id_to_word.append(word)

            # bidirectional context
            if dep_word_ind not in ind_to_context_inds:
                ind_to_context_inds[dep_word_ind] = []
            if ind not in ind_to_context_inds:
                ind_to_context_inds[ind] = []

            # we don't want to add to context prepositions
            if dep_type != "prep":
                ind_to_context_inds[dep_word_ind].append(ind)
            if dep_type != "pobj":
                ind_to_context_inds[ind].append(dep_word_ind)

            # this part is used later to collapse dependencies
            if dep_type == "prep":
                # prep(ind) <- word(dep_word_ind)
                if dep_word_ind not in ind_to_prep_dep_inds:
                    ind_to_prep_dep_inds[dep_word_ind] = []
                ind_to_prep_dep_inds[dep_word_ind].append(ind)

            if dep_type == "pobj":
                if ind not in ind_to_prep_dep_inds:
                    ind_to_prep_dep_inds[ind] = []
                ind_to_prep_dep_inds[ind].append(dep_word_ind)

            # the search is based on matching words in FIFO fashion
            if not target_ind and word == target:
                target_ind = ind

        # -1 because array indices start from 0
        # the last check is because we don't care about <eos>
        # now we're going to collect context ids and then convert them to words
        context_inds = []
        # 1. grab all words ids that the target points to
        if target_ind in ind_to_context_inds:
            context_inds += ind_to_context_inds[target_ind]

        # 2. grab all words of prep. words
        if target_ind in ind_to_prep_dep_inds:
            for prep_ind in ind_to_prep_dep_inds[target_ind]:
                for w_ind in ind_to_context_inds[prep_ind]:
                    if w_ind != target_ind:
                        context_inds.append(w_ind)
        # convert to words
        context = [id_to_word[ind-1] for ind in context_inds if len(id_to_word) > ind-1] # we don't care out <eos> dep.
        return context

    def decorate_context(self):
        tokens = self.line.split('\t')
        words = tokens[CONTEXT_TEXT_BEGIN_INDEX].split()
        words[self.target_ind] = '__'+words[self.target_ind]+'__'
        tokens[CONTEXT_TEXT_BEGIN_INDEX] = ' '.join(words)
        return '\t'.join(tokens)+"\n"     