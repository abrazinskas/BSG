'''
Base class for context sensitive inference modules
'''

import re
from nltk.stem.wordnet import WordNetLemmatizer
from jcs.data.pos import to_wordnet_pos

# just something to return in case not enough words were generated
default_generated_results = ['time', 'people', 'information', 'work', 'first', 'like', 'year', 'make', 'day', 'service']

#generated_word_re = re.compile('^[a-zA-Z]+(-[a-zA-Z]+)*$')
generated_word_re = re.compile('^[a-zA-Z]+$')


class CsInferrer(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.time = [0.0, 0]
     
     
    def inference_time(self, seconds):
        self.time[0] += seconds
        self.time[1] += 1
                   
     # processing time in msec
    def msec_per_word(self):
        return 1000*self.time[0]/self.time[1] if self.time[1] > 0 else 0.0
    
    def generate_inferred(self, result_vec, target_word, target_lemma, pos):
    
        generated_results = {}
        min_weight = None
        if result_vec is not None:
            for word, weight in result_vec:
                if generated_word_re.match(word) != None: # make sure this is not junk
                    wn_pos = to_wordnet_pos[pos]
                    lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
                    if word != target_word and lemma != target_lemma:
                        if lemma in generated_results:
                            weight = max(weight, generated_results[lemma])
                        generated_results[lemma] = weight
                        if min_weight is None:
                            min_weight = weight
                        else:
                            min_weight = min(min_weight, weight)
                            
        if min_weight is None:
            min_weight = 0.0
        i = 0.0                
        for lemma in default_generated_results:
            if len(generated_results) >= len(default_generated_results):
                break;
            i -= 1.0
            generated_results[lemma] = min_weight + i
            
                
        return generated_results
    
    
    
    def filter_inferred(self, result_vec, candidates, pos):
    
        filtered_results = {}
        candidates_found = set()
        # SO There is no way a composite word can appear?!
        if result_vec != None:

            # # TODO: this is my modification to test the difference hypothesis in our impls.
            # for word, weight in result_vec:
            #     if word in candidates:
            #         self.add_inference_result(word, weight, filtered_results, candidates_found)

            for word, weight in result_vec:
                wn_pos = to_wordnet_pos[pos]
                lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
                if lemma in candidates:
                    self.add_inference_result(lemma, weight, filtered_results, candidates_found)
                if lemma.title() in candidates:
                    self.add_inference_result(lemma.title(), weight, filtered_results, candidates_found)
                if word in candidates:  # there are some few cases where the candidates are not lemmatized
                    self.add_inference_result(word, weight, filtered_results, candidates_found)
                if word.title() in candidates:  # there are some few cases where the candidates are not lemmatized
                    self.add_inference_result(word.title(), weight, filtered_results, candidates_found)
                    
        # assign negative weights for candidates with no score
        # they will appear last sorted according to their unigram count        
#        candidates_left = candidates - candidates_found
#        for candidate in candidates_left:            
#            count = self.w2counts[candidate] if candidate in self.w2counts else 1
#            score = -1 - (1.0/count) # between (-1,-2] 
#            filtered_results[candidate] = score   
         
        return filtered_results
    
    def add_inference_result(self, token, weight, filtered_results, candidates_found):
        candidates_found.add(token)
        best_last_weight = filtered_results[token] if token in filtered_results else None
        if best_last_weight == None or weight > best_last_weight:
            filtered_results[token] = weight
        
    
    
    