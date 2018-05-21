'''
Context insensitive inferrer, based on embeddings similarities
'''

import time

from jcs.cs_inferrer import CsInferrer
from jcs.data.embedding import Embedding
from jcs.jcs_io import vec_to_str
from jcs.data.pos import to_wordnet_pos
from jcs.jcs_io import load_vocabulary_counts

from nltk.stem.wordnet import WordNetLemmatizer



class EmbeddingInferrer(CsInferrer):
    '''
    classdocs
    '''


    def __init__(self, path, vocabfile, top_inferences_to_analyze):
        CsInferrer.__init__(self)
        self.embeddings = Embedding(path)
        self.top_inferences_to_analyze = top_inferences_to_analyze
        
        self.w2counts, ignore1, ignore2 = load_vocabulary_counts(vocabfile)
        
    def new_target_key(self, target_key):
        pass
    
    def find_inferred(self, lst_instance, tfo):
        
        if lst_instance.target in self.embeddings:
            result_vec, deltatime = self.embeddings.closest_with_time(lst_instance.target, -1)
        else:
            result_vec, deltatime = None, 0
        
        tfo.write("\nDeltatime: %f msec\n" % ((deltatime)*1000))
        self.inference_time(deltatime)
            
        if (result_vec is not None):
            tfo.write("Top most similar embeddings: " + vec_to_str(result_vec, self.top_inferences_to_analyze) + '\n')
        else:
            tfo.write("Top most similar embeddings: " + " contexts: None\n") 
            
        return result_vec
            


    def filter_inferred(self, result_vec, candidates, pos):
    
        filtered_results = {}
        candidates_found = set()
        
        if result_vec != None:
            for word, weight in result_vec:
                wn_pos = to_wordnet_pos[pos]
                lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
                if lemma in candidates:
                    self.add_inference_result(lemma, weight, filtered_results, candidates_found)
                if lemma.title() in candidates: # match also capitalized words
                    self.add_inference_result(lemma.title(), weight, filtered_results, candidates_found)
                if word in candidates: # there are some few cases where the candidates are not lemmatized
                    self.add_inference_result(word, weight, filtered_results, candidates_found)
                if word.title() in candidates: # there are some few cases where the candidates are not lemmatized
                    self.add_inference_result(word.title(), weight, filtered_results, candidates_found)    
                                    
                    
        # assign negative weights for candidates with no score
        # they will appear last sorted according to their unigram count        
        candidates_left = candidates - candidates_found
        for candidate in candidates_left:            
            count = self.w2counts[candidate] if candidate in self.w2counts else 1
            score = -1 - (1.0/count) # between (-1,-2] 
            filtered_results[candidate] = score        
         
        return filtered_results
    

        