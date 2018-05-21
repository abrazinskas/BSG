from nltk.corpus import wordnet

to_wordnet_pos = {'N':wordnet.NOUN,'J':wordnet.ADJ,'V':wordnet.VERB,'R':wordnet.ADV}
from_lst_pos = {'j':'J','a':'J', 'v':'V', 'n':'N', 'r':'R'}