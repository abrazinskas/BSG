'''
Context-sensitive inference model based on syntactic dependency embeddings
Used in the paper:
A Simple Word Embedding Model for Lexical Substitution. Workshop on Vector Space Modeling for NLP (VSM), 2015 (VSM 2015)
'''

from cs_inferrer import CsInferrer
from jcs.jcs_io import load_vocabulary_counts
from jcs.data.conll_line import ConllLine
from jcs.jcs_io import vec_to_str
from jcs.data.pos import to_wordnet_pos
from jcs.data.embedding import Embedding
import numpy as np
import sys

from nltk.stem.wordnet import WordNetLemmatizer

    
def read_conll(conll_file, lower):
    root = ConllLine()
    words = [root]
    for line in conll_file:
        line = line.strip()
        if len(line) > 0:
            if lower: 
                line = line.lower()
            tokens = line.split('\t')
            words.append(ConllLine(tokens))
        else:
            if len(words)>1: 
                yield words
                words = [root]
    if len(tokens) > 1:
        yield tokens

    
def get_deps(sent, target_ind, stopwords):
    
    deps = []
      
    for word_line in sent[1:]:
        parent_line = sent[word_line.head]
#universal        if word_line.deptype == 'adpmod': # we are collapsing preps 
        if word_line.deptype == 'prep': # we are collapsing preps
            continue 
#universal       if word_line.deptype == 'adpobj' and parent_line.id != 0: # collapsed dependency
        if word_line.deptype == 'pobj' and parent_line.id != 0: # collapsed dependency
            grandparent_line = sent[parent_line.head]
            if (grandparent_line.id != target_ind and word_line.id != target_ind):
                continue
            relation = "%s:%s" %  (parent_line.deptype, parent_line.form)
            head = grandparent_line.form
        else: # direct dependency
            if (parent_line.id != target_ind and word_line.id != target_ind):
                continue
            head = parent_line.form
            relation = word_line.deptype
        if word_line.id == target_ind:
            if head not in stopwords:
                deps.append("I_".join((relation,head)))
        else:
            if word_line.form not in stopwords:
                deps.append("_".join((relation,word_line.form)))
#      print h,"_".join((rel,m))
#      print m,"I_".join((rel,h))
    return deps



class CsEmbeddingInferrer(CsInferrer):

    def __init__(self, vocabfile, ignore_target, context_math, word_path, context_path, conll_filename, window_size, top_inferences_to_analyze):
        
        CsInferrer.__init__(self)
        self.ignore_target = ignore_target
        self.context_math = context_math
        self.word_vecs = Embedding(word_path)
        self.context_vecs = Embedding(context_path) 
        self.use_stopwords = False
        
        assert(not (window_size >= 0 and conll_filename is not None))
        self.window_size = window_size
        if (window_size < 0):
            self.conll_file = open(conll_filename, 'r')       
            self.sents = read_conll(self.conll_file, True) # this is generator as far as I can tell
        self.top_inferences_to_analyze = top_inferences_to_analyze
        
        self.w2counts, _, self.stopwords = load_vocabulary_counts(vocabfile)

#    def close(self):
#        self.conll_file.close()
        
    def represent(self, target, deps, avg_flag, tfo):
        
        target_vec = None if target is None else np.copy(self.word_vecs.represent(target))
        dep_vec = None
        deps_found = 0
        for dep in deps:
            if dep in self.context_vecs:
                deps_found += 1
                if dep_vec is None:
                    dep_vec = np.copy(self.context_vecs.represent(dep))
                else:
                    dep_vec += self.context_vecs.represent(dep)                
            else:
                tfo.write("NOTICE: %s not in context embeddings. Ignoring.\n" % dep)
        
        ret_vec = None
        if target_vec is not None:
            ret_vec = target_vec
        if dep_vec is not None:
            if avg_flag:
                dep_vec /= deps_found
            if ret_vec is None:
                ret_vec = dep_vec
            else:
                ret_vec += dep_vec
        
        norm = (ret_vec.dot(ret_vec.transpose()))**0.5
        ret_vec /= norm
        
        return ret_vec
    
    def mult(self, target, deps, geo_mean_flag, tfo):
        
        #SUPPORT NONE TARGET
        
        target_vec = self.word_vecs.represent(target)
        scores = self.word_vecs.pos_scores(target_vec) # performs dot product with the whole vocabulary
        for dep in deps:
            if dep in self.context_vecs:
                dep_vec = self.context_vecs.represent(dep)
                mult_scores = self.word_vecs.pos_scores(dep_vec) # same here: performs dot product with the whole vocabulary
                if geo_mean_flag:
                    mult_scores = mult_scores**(1.0/len(deps)) # TODO: Here he forgets that he should compute the number of seen context words
                scores = np.multiply(scores, mult_scores)
            else:
                tfo.write("NOTICE: %s not in context embeddings. Ignoring.\n" % dep)   
                
        result_vec = self.word_vecs.top_scores(scores, -1)                
        return result_vec
            
    
    
    
    def extract_contexts(self, lst_instance):
        # performs context context extraction with dependency parse input
        if self.window_size < 0:
            cur_sent = next(self.sents)
            cur_sent_target_ind = lst_instance.target_ind+1
            # Why does he loop like so? Can't he just lookup instead?
            # seems like it's first match loop
            while ((cur_sent_target_ind < len(cur_sent) and cur_sent[cur_sent_target_ind].form != lst_instance.target)):
                sys.stderr.write("Target word form mismatch in target id %s: %s != %s  Checking next word.\n" % (lst_instance.target_id, cur_sent[cur_sent_target_ind].form, lst_instance.target))
                cur_sent_target_ind += 1
            if cur_sent_target_ind == len(cur_sent):
                sys.stderr.write("Start looking backwards.\n")
                cur_sent_target_ind = lst_instance.target_ind
                while ((cur_sent_target_ind > 0) and (cur_sent[cur_sent_target_ind].form != lst_instance.target) ):
                    sys.stderr.write("Target word form mismatch in target id %s: %s != %s  Checking previous word.\n" % (lst_instance.target_id, cur_sent[cur_sent_target_ind].form, lst_instance.target))
                    cur_sent_target_ind -= 1
            if  cur_sent_target_ind == 0:
                sys.stderr.write("ERROR: Couldn't find a match for target.")
                cur_sent_target_ind = lst_instance.target_ind+1
            stopwords = self.stopwords if self.use_stopwords else set()
            contexts = get_deps(cur_sent, cur_sent_target_ind, stopwords)
        else:
            contexts = lst_instance.get_neighbors(self.window_size)
            
        
        return contexts
            
        
    def find_inferred(self, lst_instance, tfo):
                
        contexts = self.extract_contexts(lst_instance) # at this stage we're grabbing contexts from DP or via window!!!!
        tfo.write("Contexts for target %s are: %s\n" % (lst_instance.target, contexts))        
        contexts = [c for c in contexts if c in self.context_vecs]
        tfo.write("Contexts in vocabulary for target %s are: %s\n" % (lst_instance.target, contexts))
        if self.ignore_target:
            target = None
        else:    
            if lst_instance.target not in self.word_vecs:
                tfo.write("ERROR: %s not in word embeddings.Trying lemma.\n" % lst_instance.target)
                if lst_instance.target_lemma not in self.word_vecs:
                    tfo.write("ERROR: lemma %s also not in word embeddings. Giving up.\n" % lst_instance.target_lemma)
                    return None
                else:
                    target = lst_instance.target_lemma
            else:
                target = lst_instance.target
          
        # 'add' and 'avg' metrics are implemented more efficiently with vector representation arithmetics
        # as shown in Omer's linguistic regularities paper, this is equivalent as long as the vectors are normalized to 1                         
        if self.context_math == 'add':
            cs_rep = self.represent(target, contexts, False, tfo)
            if cs_rep is None:
                cs_rep = self.word_vecs.zeros() 
            result_vec = self.word_vecs.closest_vec(cs_rep, -1)
        elif self.context_math == 'avg':
            cs_rep = self.represent(target, contexts, True, tfo)
            if cs_rep is None:
                cs_rep = self.word_vecs.zeros() 
            result_vec = self.word_vecs.closest_vec(cs_rep, -1)
        elif self.context_math == 'mult':
            result_vec = self.mult(target, contexts, False, tfo)
        elif self.context_math == 'geomean':
            result_vec = self.mult(target, contexts, True, tfo)
        elif self.context_math == 'none' and self.ignore_target is not None:
            result_vec = self.word_vecs.closest(target, -1)
        else:
            raise Exception('Unknown context math: %s' % self.context_math)
            
                
            
        if (result_vec is not None):
            tfo.write("Top most similar embeddings: " + vec_to_str(result_vec, self.top_inferences_to_analyze) + '\n')
        else:
            tfo.write("Top most similar embeddings: " + " contexts: None\n") 
            
        return result_vec
            

