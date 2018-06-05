import os
from libraries.evaluation.entailment.entailment import test_entailment, test_directional_entailment
from libraries.evaluation.word_sim.all_wordsim import all_word_sim
from libraries.evaluation.word_sim.wordsim import word_sim
from libraries.evaluation.GloVe.evaluate import glove_evaluate


# evaluates vectors on Glove's benchmark and offline wordvectors.org benchmark
# vectors_path : filename or a folder with vectors
# vocab: vocab object, should be passed as there is currently a circular dependency TODO: fix!
def evaluate(mu_vectors_files, sigma_vectors_files=None, vocab_file=None, vocab=None,
             max_count=None, full_sim=False, log_sigmas=False):

    # all similarity tests are performed on mu vectors
    for mu_vectors_file in mu_vectors_files:
        # https://github.com/mfaruqui/eval-word-vectors
        # 1. similarity tests
        if full_sim:
            sim_folder = os.path.dirname(os.path.realpath(__file__))+"/word_sim/data/word-sim"
            # sim_folder = os.path.join(os.getcwd(), "../evaluation/word_sim/data/word-sim")
            all_word_sim(mu_vectors_file, sim_folder)
        else:
            all_sim_file = os.path.dirname(os.path.realpath(__file__))+"/word_sim/data/combined-word-sim/TEST.txt"
            # all_sim_file = os.path.join(os.getcwd(), "/../evaluation/word_sim/data/combined-word-sim/TEST.txt")
            word_sim(mu_vectors_file, all_sim_file)
        # https://github.com/stanfordnlp/GloVe

        # 2. analogical reasoning
        if vocab_file is not None:
            glove_evaluate(vocab_file, mu_vectors_file, max_count=max_count)
    if not sigma_vectors_files:
        return

    for mu_vectors_file, sigma_vectors_file in zip(mu_vectors_files, sigma_vectors_files):
        # print "mu_vectors_file: %s" % mu_vectors_file
        # print "sigma_vectors_file: %s"% sigma_vectors_file

        # 3. KL entailment
        for sf in ["kl", "cos", "l2"]:
            test_entailment(mu_vectors_path=mu_vectors_file, sigma_vectors_path=sigma_vectors_file, log_sigmas=log_sigmas,
                            score_func=sf, normalize=False)

        # 4. directional entailment on Baroni
        test_directional_entailment(mu_vectors_path=mu_vectors_file, sigma_vectors_path=sigma_vectors_file,
                                    test_path='/data/bench/baroni2012_dir/data.tsv', header=True, vocab=vocab,
                                    log_sigmas=log_sigmas)
        # 5. directional entailment on Bless
        test_directional_entailment(mu_vectors_path=mu_vectors_file, sigma_vectors_path=sigma_vectors_file,
                                    test_path='/data/bench/bless2011_dir/data.tsv', header=True, vocab=vocab,
                                    log_sigmas=log_sigmas)