import sys
from conll_line import ConllLine

for tree_line in sys.stdin:
    tree_line_stripped = tree_line.strip()
    if len(tree_line_stripped) > 0:        
        conll = ConllLine()
        conll.from_tree_line(tree_line_stripped)
        sys.stdout.write(str(conll)+'\n')
    else:
        sys.stdout.write('\n')