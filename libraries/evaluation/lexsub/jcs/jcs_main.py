'''
Run lexical substitution experiments
'''
import sys
import time
import argparse
import re
import numpy

from jcs.jcs_io import extract_word_weight

from jcs.data.context_instance import ContextInstance


from jcs.jcs_io import vec_to_str
from jcs.jcs_io import vec_to_str_generated
from jcs.cs_embedding_inferrer import CsEmbeddingInferrer
from jcs.embedding_inferrer import EmbeddingInferrer
# from jcs.context2vec_inferrer import Context2vecInferrer

target_re = re.compile(".*__(.*)__.*")

        
def read_candidates(candidates_file):
    target2candidates = {}
    # finally.r::eventually;ultimately
    with open(candidates_file, 'r') as f:
        for line in f:
            segments = line.split('::')
            target = segments[0]
            candidates = set(segments[1].strip().split(';'))
            target2candidates[target] = candidates
    return target2candidates
        
def run_test(inferrer):
    
    if args.candidatesfile != None:
        target2candidates = read_candidates(args.candidatesfile)
    else:
        target2candidates = None

    tfi = open(args.testfile, 'r')
    tfo = open(args.resultsfile, 'w')
    tfo_ranked = open(args.resultsfile+'.ranked', 'w')
    # tfo_generated_oot = open(args.resultsfile+'.generated.oot', 'w')
    # tfo_generated_best = open(args.resultsfile+'.generated.best', 'w')
    
    lines = 0
    while True:
        context_line = tfi.readline()
        if not context_line:
            break
        lst_instance = ContextInstance(context_line, args.no_pos)
        lines += 1
        if (args.debug == True):
            tfo.write("\nTest context:\n")
            tfo.write("***************\n")
            
        tfo.write(lst_instance.decorate_context())
        if lines == 159:
            print 'test'

        # performs some operation(dot product) over all words and returns a score for each word
        result_vec = inferrer.find_inferred(lst_instance, tfo)

        generated_results = inferrer.generate_inferred(result_vec, lst_instance.target, lst_instance.target_lemma, lst_instance.pos)
        
        tfo.write("\nGenerated lemmatized results\n")
        tfo.write("***************\n")
        tfo.write("GENERATED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(generated_results.iteritems(), args.topgenerated)+"\n")
        # tfo_generated_oot.write(' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(generated_results.iteritems(), args.topgenerated)+"\n")
        # tfo_generated_best.write(' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " :: " + vec_to_str_generated(generated_results.iteritems(), 1)+"\n")
        
        filtered_results = inferrer.filter_inferred(result_vec, target2candidates[lst_instance.target_key], lst_instance.pos)
        
        tfo.write("\nFiltered results\n")
        tfo.write("***************\n")
        tfo.write("RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(filtered_results.iteritems(), len(filtered_results))+"\n")
        tfo_ranked.write("RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(filtered_results.iteritems(), len(filtered_results))+"\n")
        
#        print "end %f" % time.time()
        
        if lines % 10 == 0:
            print "Read %d lines" % lines                      
        
    print "Read %d lines in total" % lines 
    print "Time per word: %f msec" % inferrer.msec_per_word()          
    tfi.close()
    tfo.close()
    tfo_ranked.close()
    # tfo_generated_oot.close()
    # tfo_generated_best.close()
    
def run(args):
    
    print time.asctime(time.localtime(time.time()))
    print args.embeddingpath
    print args.contextmath
    if args.inferrer == 'emb':    
        inferrer = CsEmbeddingInferrer(args.vocabfile, args.ignoretarget, args.contextmath, args.embeddingpath, args.embeddingpathc, args.testfileconll, args.bow_size, 10)
        print "Using CsEmbeddingInferrer"
    
    elif args.inferrer == 'lstm':
        raise NotImplementedError
        # inferrer = Context2vecInferrer(args.lstm_config, args.ignoretarget, args.contextmath, 10)
        print "Using Context2vecInferrer"
    elif args.inferrer == 'none':
        inferrer = EmbeddingInferrer(vocabfile=args.vocabfile, path=args.embeddingpath, top_inferences_to_analyze=10)
        print 'using context insensitive inferrer'

    else:
        raise Exception("Unknown inferrer type: " + args.inferrer)
    
    print time.asctime(time.localtime(time.time()))
    
    run_test(inferrer)
    print "Finished"
    print time.asctime(time.localtime(time.time()))


    
if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='JCS utility')
    
    
    parser.add_argument('--inferrer', choices=['lstm', 'emb', 'none'],
                    default='lstm',
                    help='context type ("lstm", "emb")')
    
    # Only for Context2vecInferrer
    parser.add_argument('-lstm_config', action="store", dest="lstm_config", default=None, help="config file of lstm context model and respective word embeddings")
    
    # Only for CsEmbeddingInferrer
    parser.add_argument('-embeddingpath', action="store", dest="embeddingpath", default=None, help="prefix to files containing word embeddings")
    parser.add_argument('-embeddingpathc', action="store", dest="embeddingpathc", default=None, help="prefix to files containing context word embeddings")
    parser.add_argument('-vocabfile', action="store", dest="vocabfile")
    parser.add_argument('-bow',action='store',dest='bow_size', default=-1, type=int, help="context bag-of-words window size.  0 means entire sentence. -1 means syntactic dependency contexts.")
    
    # Common
    parser.add_argument('-targetsfile', action="store", dest="targetsfile", default=None)
    parser.add_argument('-testfile', action="store", dest="testfile")
    parser.add_argument('-testfileconll', action="store", dest="testfileconll", default=None, help="test file with sentences parsed in conll format")
    parser.add_argument('-candidatesfile', action="store", dest="candidatesfile", default=None)
    parser.add_argument('-resultsfile', action="store", dest="resultsfile")
    parser.add_argument('-contextmath', action="store", dest="contextmath", default=None, help="arithmetics used to consider context [add|mult|geomean|none]")
    parser.add_argument('--ignoretarget', action="store_true", dest="ignoretarget", default=False, help="ignore lhs target. compute only context compatibility.")
    parser.add_argument('--nopos',action='store_true',dest='no_pos', default=False, help="ignore part-of-speech of target word")
    parser.add_argument('-topgenerated', action="store", dest="topgenerated", type=int, default=10, help="top entries to print in generated parvecs")

    parser.add_argument('--debug',action='store_true',dest='debug')

    
    args = parser.parse_args(sys.argv[1:])
    
    config_file_name = args.resultsfile + ".CONFIG"
    cf = open(config_file_name, 'w')
    cf.write(' '.join(sys.argv)+'\n')
    cf.close()
    
    numpy.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
    
    run(args)
    
