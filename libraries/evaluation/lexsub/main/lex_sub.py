import os, sys
import operator
from support import read_candidates, wf2ws, flatten, get_best_scores_for_candidates, conll_skip_sentence
from context_instance import ContextInstance
import numpy as np
from libraries.utils.paths_and_files import create_folders_if_not_exist
from libraries.evaluation.lexsub.jcs.evaluation.lst.lst_gap import compute_gap


def lex_sub(embeddings, input_type, target_words_vocab, context_words_vocab, output_path, candidates_file, conll_file,
            test_file, gold_file, half_window_size=5, arithm_type=None):
    conll = None
    if input_type == "dependency":
        conll = open(conll_file, "r")

    # this is a very time consuming part
    all_candidates = read_candidates(candidates_file, allowed_words=target_words_vocab)  # read candidates
    ranked_file_path = os.path.join(output_path, "ranked_bsg.txt")
    create_folders_if_not_exist(ranked_file_path)
    output = open(ranked_file_path, "w")
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            # if i % 1 == 0:
            #     print "----------------------"
            #     print 'reading line # %d' % (i+1)
            lst_instance = ContextInstance(line)

            # find if the target(center) word in in the vocab ( either his lemma or word)
            target = None
            if lst_instance.target in target_words_vocab:
                target = lst_instance.target
            elif lst_instance.target_lemma in target_words_vocab:
                target = lst_instance.target_lemma

            # sometimes the target(center) word can be not in vocab, so we skip that instance
            if target and lst_instance.target_key in all_candidates:
                left_context, right_context = lst_instance.get_neighbors(half_window_size) if input_type == "normal" else\
                    lst_instance.get_dep_context(conll, lst_instance.target)

                # perform filtering by throwing away all words that do not appear in vocab
                # I do it after creating windows because of indexing problem
                left_context = [c for c in left_context if c in context_words_vocab]
                right_context = [c for c in right_context if c in context_words_vocab]

                # print "---------------------------------------"
                # print "left context for the word \"%s\" is : %s" % (target, str(left_context))
                # print "right context for the word \"%s\" is : %s" % (target, str(right_context))

                # grab candidates
                candidates = all_candidates[lst_instance.target_key]  # note that candidates is a dictionary

                # format candidates properly
                formatted_candidates = np.unique(flatten(candidates.values()))

                # rank them
                scores = embeddings.score(target=target, left_context=left_context, right_context=right_context,
                                          candidates=formatted_candidates, ar_type=arithm_type)

                # print(candidates.items())
                # perform mapping to all candidates
                scores = get_best_scores_for_candidates(candidates, scores)

                # sort in descending order
                sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
                formatted_scores = [' '.join([word, wf2ws(score)]) for word, score in sorted_scores]
                formatted_scores = "\t" + "\t".join(formatted_scores)
            else:
                formatted_scores = ""
                # print 'center word %s is not in the vocabulary' %(target)
                # print "skipping start"
                if input_type == "dependency":
                    conll_skip_sentence(conll)  # to make sure that pointers are aligned
                # print "skipping end"

            # write to a file
            output.write("RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + formatted_scores + "\n")
    if conll:
        conll.close()
    output.close()

    # compute gap
    compute_gap(gold_file_path=gold_file, eval_file_path=ranked_file_path, out_file_path=os.path.join(output_path, "gap.txt"),
                ignore_mwe=True)
