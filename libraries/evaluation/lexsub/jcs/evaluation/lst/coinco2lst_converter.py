'''
Used to convert the coinco (Kremer 2014) xml dataset format to the format used with LST 2007.

Example of coinco format:
<document>
  <sent MASCfile="NYTnewswire9.txt" MASCsentID="s-r0" >
    <precontext>
    
    </precontext>
    <targetsentence>
    A mission to end a war
    </targetsentence>
    <postcontext>
    AUSTIN, Texas -- Tom Karnes was dialing for destiny, but not everyone wanted to cooperate.
    </postcontext>
    <tokens>
      <token id="XXX" wordform="A" lemma="a" posMASC="XXX" posTT="DT" />
      <token id="4" wordform="mission" lemma="mission" posMASC="NN" posTT="NN" problematic="no" >
        <substitutions>
          <subst lemma="calling" pos="NN" freq="1" />
          <subst lemma="campaign" pos="NN" freq="1" />
          <subst lemma="dedication" pos="NN" freq="1" />
          <subst lemma="devotion" pos="NN" freq="1" />
          <subst lemma="duty" pos="NN" freq="1" />
          <subst lemma="effort" pos="NN" freq="1" />
          <subst lemma="goal" pos="NN" freq="2" />
          <subst lemma="initiative" pos="NN" freq="1" />
          <subst lemma="intention" pos="NN" freq="1" />
          <subst lemma="movement" pos="NN" freq="1" />
          <subst lemma="plan" pos="NN" freq="2" />
          <subst lemma="pursuit" pos="NN" freq="1" />
          <subst lemma="quest" pos="NN" freq="1" />
          <subst lemma="step" pos="NN" freq="1" />
          <subst lemma="task" pos="NN" freq="2" />
        </substitutions>
      </token>

'''

import sys
import string
from operator import itemgetter
from xml.etree import ElementTree


def is_printable(s):
    return all(c in string.printable for c in s)


def clean_token(token):
    
    token = token.replace('&quot;', '"')
    token = token.replace('&apos;', "'")
    token = token.replace(chr(int("85",16)), "...")
    token = token.replace(chr(int("91",16)), "'")
    token = token.replace(chr(int("92",16)), "'")
    token = token.replace(chr(int("93",16)), '"')
    token = token.replace(chr(int("94",16)), '"')
    token = token.replace(chr(int("96",16)), '-') 
        
    if not is_printable(token):
        sys.stderr.write('TOKEN NOT PRINTABLE: '+''.join([str(c) for c in token if c in string.printable ]) + '\n')
        return "<UNK>"
    else:
        return token

def subs2text(subs_element):
    subs = [(int(sub.attrib.get('freq')), sub.attrib.get('lemma').replace(';', ',')) for sub in subs_element.iter('subst')]  # sub.attrib.get('lemma').replace(';', ',') is used to fix a three cases in coinco where the lemma includes erroneously the char ';'. Since this char is used as a delimiter, we replace it with ','. 
    sorted_subs = sorted(subs, reverse=True)
    return ';'.join([sub + " " + str(freq) for freq, sub in sorted_subs])+';'

if __name__ == '__main__':
    
    if len(sys.argv)<3:
        print "Usage: %s <coinco-filename> <output-vocab-filename> <output-test-filename> <output-gold-filename>" % sys.argv[0]
        sys.exit(1)
        
    with open(sys.argv[1], 'r') as f:
        coinco = ElementTree.parse(f)
    
    vocab_file = open(sys.argv[2], 'w') # targets vocabulary helper file (not part of the LST 2007 format)   
    test_file = open(sys.argv[3], 'w') # LST 2007 test format
    gold_file = open(sys.argv[4], 'w') # LST 2007 gold format
     
    pos_types = set()
    target_types = {}   
    wordforms = {}
    sent_num = 0
    tokens_num = 0
        
    for sent in coinco.iter('sent'):
        sent_num += 1
        tokens = sent.find('tokens')
        sent_text = ""
        for token in tokens.iter('token'):
            sent_text = sent_text + clean_token(token.attrib.get('wordform')).lower() + " "                
            if token.attrib.get('id') != 'XXX':
                wordform = token.attrib.get('wordform').lower()
                if not '-' in wordform:
                    if wordform in wordforms:
                        wordforms[wordform] = wordforms[wordform]+1
                    else:
                        wordforms[wordform] = 1
        sent_text = sent_text.strip()
        tok_position = -1
        for token in tokens.iter('token'):
            tok_position += 1
            if token.attrib.get('id') != 'XXX' and token.attrib.get('problematic') == 'no':# and (len(token.attrib.get('lemma').strip().split()) == 1):
                tokens_num += 1
                try:
                    target_key = clean_token(token.attrib.get('lemma')) + '.' + token.attrib.get('posMASC')[0]
                    test_file.write("%s\t%s\t%d\t%s\n" % (target_key, token.attrib.get('id'), tok_position, sent_text))
                    gold_file.write("%s %s :: %s\n" % (target_key, token.attrib.get('id'), subs2text(token.find('substitutions'))))
                    
#                    pos_types.add(token.attrib.get('posMASC'))
#                    if token.attrib.get('posMASC').startswith('V') and token.attrib.get('problematic') == 'no':
#                        if target_key in target_types:
#                            target_types[target_key] += 1
#                        else:
#                            target_types[target_key] = 1
                    
                except UnicodeEncodeError as e:
                    test_file.write("ENCODING TARGET ERROR at token_id %s\n" % (token.attrib.get('id')))
                    gold_file.write("ENCODING TARGET ERROR at token_id %s\n" % (token.attrib.get('id')))
                    sys.stderr.write("ENCODING TARGET ERROR at token_id %s\n" % (token.attrib.get('id')))
                
                
    sorted_wordforms = sorted(wordforms.iteritems(), key=itemgetter(1), reverse=True)
    
    for wordform, freq in sorted_wordforms:
        try:
            vocab_file.write('%s.*\t%d\n' % (wordform, freq))
        except UnicodeEncodeError as e:
            sys.stderr.write("error encoding\n")

    vocab_file.close()
    test_file.close()
    gold_file.close()
                           
    print 'read %d sentences %d target tokens' % (sent_num, tokens_num)   
#    print 'target types with >= 10 instances: %d' % len([t for (t, freq) in target_types.iteritems() if freq>=10])    
    
    
        
    