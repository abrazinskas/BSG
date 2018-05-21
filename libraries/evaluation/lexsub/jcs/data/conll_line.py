import re

class ConllLine():
 


    def root_init(self):
        self.id = 0
        self.form = '*root*'
        self.lemma = '_'
        self.cpostag = '_'
        self.postag = '_'
        self.feats = '_'
        self.head = -1
        self.deptype = 'rroot'
        self.phead = -1 
        self.pdeptype = '_'

    def __str__( self ):
        return '\t'.join([str(self.id), self.form, self.lemma, self.cpostag, self.postag, self.feats, str(self.head), self.deptype, str(self.phead), self.pdeptype])
    
    def __init__(self, tokens=None):
        if tokens == None:
            self.root_init()
        else:
            self.id = int(tokens[0])
            self.form = tokens[1]
            self.lemma = tokens[2]
            self.cpostag = tokens[3]
            self.postag = tokens[4]
            self.feats = tokens[5]
            self.head = int(tokens[6])
            self.deptype = tokens[7]
            if len(tokens) > 8:
                self.phead = -1 if tokens[8] == '_' else int(tokens[8]) 
                self.pdeptype = tokens[9]
            else:
                self.phead = -1 
                self.pdeptype = '_'
        
    tree_line_extractor = re.compile('([a-z]+)\(.+-(\d+), (.+)-(\d+)\)') 
    # stanford parser tree output:  num(Years-3, Five-1) 
    def from_tree_line(self, tree_line):
        self.root_init()
        tok = self.tree_line_extractor.match(tree_line)
        self.id = int(tok.group(4))
        self.form = tok.group(3)
        self.head = int(tok.group(2))
        self.deptype = tok.group(1)
        

        
