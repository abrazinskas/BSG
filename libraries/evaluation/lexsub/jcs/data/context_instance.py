'''
Instance in the Lexical Substitution Task dataset

'''


from pos import from_lst_pos

CONTEXT_TEXT_BEGIN_INDEX = 3
TARGET_INDEX = 2




class ContextInstance(object):
 
    def __init__(self, line, no_pos_flag):
        '''
        Constructor
        '''
        self.line = line
        tokens1 = line.split("\t")
        self.target_ind = int(tokens1[TARGET_INDEX])
        self.words = tokens1[3].split()
        self.target = self.words[self.target_ind]       
        self.full_target_key = tokens1[0]
        self.pos = self.full_target_key.split('.')[-1]
        self.target_key = '.'.join(self.full_target_key.split('.')[:2]) # remove suffix in cases of bar.n.v
        self.target_lemma = self.full_target_key.split('.')[0]       
        self.target_id = tokens1[1]

        # I don't see why I need this one?
        # if self.pos in from_lst_pos:
        #     self.pos = from_lst_pos[self.pos]

        self.target_pos = '.'.join([self.target, '*']) if no_pos_flag == True else '.'.join([self.target, self.pos])
   
    def get_neighbors(self, window_size):
        tokens = self.line.split()[3:]
        
        if (window_size > 0):                                    
            start_pos = max(self.target_ind-window_size, 0)
            end_pos = min(self.target_ind+window_size+1, len(tokens))
        else:
            start_pos = 0
            end_pos = len(tokens)
            
        neighbors = tokens[start_pos:self.target_ind] + tokens[self.target_ind+1:end_pos]
        return neighbors 
   
    def decorate_context(self):
        tokens = self.line.split('\t')
        words = tokens[CONTEXT_TEXT_BEGIN_INDEX].split()
        words[self.target_ind] = '__'+words[self.target_ind]+'__'
        tokens[CONTEXT_TEXT_BEGIN_INDEX] = ' '.join(words)
        return '\t'.join(tokens)+"\n"     