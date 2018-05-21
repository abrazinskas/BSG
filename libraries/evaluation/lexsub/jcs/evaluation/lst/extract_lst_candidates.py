'''
Used to extract (or pool) substitute candidates for every target type in the LST dataset
'''
import sys

if __name__ == '__main__':
    
    if len(sys.argv)<3:
        print "Usage: %s <lst-gold-file> <candidates-file> [no-mwe]" % sys.argv[0]
        sys.exit(1)
        
    goldfile = open(sys.argv[1], 'r')
    outfile = open(sys.argv[2], 'w')
    
    ignore_mwe = False
    if (len(sys.argv) > 3):
        sys.stderr.write("ignoring multi-word-expressions\n");
        ignore_mwe = True        
    
    good_oneword_inst = 0
    target2candidates = {}
    # bright.a 5 :: intelligent 3;clever 2;most able 1;capable 1;promising 1;sharp 1;motivated 1;
    for line in goldfile:
        if len(line)>0:
            oneword_in_line = 0 # e.g. ;most able 1;
            segments = line.split("::")
            if len(segments)>=2:
                target = segments[0][:segments[0].strip().rfind(' ')]
                target = '.'.join(target.split('.')[:2]) # remove suffix in cases of bar.n.v
                line_candidates = segments[1].strip().split(';')
                for candidate_count in line_candidates:
                    if len(candidate_count) > 0:
                        delimiter_ind = candidate_count.rfind(' ')
                        candidate = candidate_count[:delimiter_ind]
                        if ignore_mwe and ((len(candidate.split(' '))>1) or (len(candidate.split('-'))>1)):
                            continue
                        oneword_in_line += 1                       
                        if target in target2candidates:
                            candidates = target2candidates[target]
                        else:
                            candidates = set()
                            target2candidates[target] = candidates
                        candidates.add(candidate)
            if (oneword_in_line >= 1):
                good_oneword_inst += 1
    
    if ignore_mwe:
        sys.stderr.write("good_oneword_inst: " + str(good_oneword_inst) + "\n")        
    for target, candidates in target2candidates.iteritems():
        outfile.write(target + '::' + ';'.join(list(candidates)) + '\n')
    
    goldfile.close()
    outfile.close()
        