import os
import glob
import numpy as np
from support import get_kl_scores, get_cos_scores, get_l2_scores, read_vectors_to_dict, KL, cosine_sim, variance
# import matplotlib.pyplot as plt


# returns both precision and accuracy
# mus and sigmas are dictionaries
def test_entailment(mu_vectors_path, sigma_vectors_path, test_path='/data/bench/baroni2012_one/', plot=False,
                    score_func="kl", log_sigmas=True, kl_type="gauss", normalize=True):
    assert kl_type in ["gauss", "vMF"]
    assert score_func in ["kl", "cos", 'l2']
    # read test files
    mus_and_sigmas = read_vectors_to_dict(mu_vectors_path, sigma_vectors_path, log_sigmas=log_sigmas)
    prefix = os.path.dirname(os.path.realpath(__file__))
    test_path = prefix + test_path
    if os.path.isdir(test_path):
        filenames = glob.glob(test_path + "/*")
    else:
        filenames = [test_path]  # that means there is only one file
    for filename in filenames:
            if score_func == "kl":
                scores_and_ent, seen, total, words = get_kl_scores(mus_and_sigmas, filename, kl_type=kl_type,
                                                                   normalize=normalize)
            if score_func == "cos":
                scores_and_ent, seen, total, words = get_cos_scores(mus_and_sigmas, filename)
            if score_func == "l2":
                scores_and_ent, seen, total, words = get_l2_scores(mus_and_sigmas, filename, normalize=normalize)

            ent_true = scores_and_ent[scores_and_ent[:, 1] == 1, 0]
            ent_false = scores_and_ent[scores_and_ent[:, 1] == 0, 0]

            opt_th, pred, PR, R, F1 = __fit_th(scores_and_ent, score_func=score_func)
            print " ---------------------------------------------------- "
            print " ------------- NON-DIRECTIONAL ENTAILMENT TEST -------------"
            print "------------- %s -------------" % filename
            print "scoring function: %s " % score_func
            print "pos/neg ratio: %f" % (float(np.sum(scores_and_ent[:, 1] == 1))/float(np.sum(scores_and_ent[:,1]==0)))
            print "PR is: %f, R is: %f, F1 is: %f " %(PR, R, F1)
            print "seen %f %% " % ((float(seen)/float(total)) * 100.0)
            print "optimal th is %f " % opt_th
            if plot:
                bins = 50
                his1, _ = np.histogram(ent_true, bins=bins)
                his2, _ = np.histogram(ent_false, bins=bins)
                max_count = np.max(his1 + his2)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.hist(ent_true, bins=bins, alpha=0.5, label="entail")
                ax1.hist(ent_false, bins=bins, alpha=0.5, label="not entail")
                ax1.set_xlabel(score_func)
                ax1.set_ylabel("count")
                y_line = np.linspace(0, max_count, 100)
                ax1.plot(np.repeat([opt_th],100),y_line, 'r')
                ax1.set_title("F1("+score_func+") is "+str(+round(F1,3)))
                plt.legend(loc="upper right")
                plt.show()

# fits the optimal threshold based on F1 score
def __fit_th(scores_and_ent, max_iter=100000, score_func="kl"):
    step = (max(scores_and_ent[:, 0]) - min(scores_and_ent[:, 0]))/float(max_iter)
    th = min(scores_and_ent[:, 0])
    ths = []
    f1s = []
    for i in range(max_iter):
        # s = scores_and_ent[:, 0] < th if score_func == "KL" else scores_and_ent[:, 0] > th
        # corr = np.sum(s == scores_and_ent[:, 1])
        pred = scores_and_ent[:, 0] > th if score_func == "cos" else scores_and_ent[:, 0] < th
        _, _ , f1 = __compute_PR_R_F1(scores_and_ent, pred)
        ths.append(th)
        f1s.append(f1)
        th += step
    max_f1_idx = f1s.index(max(f1s))
    opt_th = ths[max_f1_idx]
    # TODO avoid extra call
    pred = scores_and_ent[:, 0] > opt_th if score_func == "cos" else scores_and_ent[:, 0] < opt_th
    PR, R, F1 = __compute_PR_R_F1(scores_and_ent, pred)
    return opt_th, pred, PR, R, F1

def __compute_PR_R_F1(scores_and_ent, pred):
    TP = np.sum((pred == 1) * (scores_and_ent[:, 1] == 1))
    TN = np.sum((pred == 0) * (scores_and_ent[:, 1] == 0))
    FP = np.sum((pred == 1) * (scores_and_ent[:, 1] == 0))
    FN = np.sum((pred == 0) * (scores_and_ent[:, 1] == 1))
    if TP == 0:
        return 0, 0, 0
    PR = float(TP)/float(TP+FP)
    R = float(TP) / float(TP + FN)
    F1 = (2*PR*R)/(PR+R)
    return PR, R, F1



# if vocabulary is passed, it will check what would be direction assignments based purely on frequency
def test_directional_entailment(mu_vectors_path, sigma_vectors_path, header=False,
                                test_path='data/bench/baroni2012_dir/data.tsv', debug=False, vocab=None, log_sigmas=True):
    mus_and_sigmas = read_vectors_to_dict(mu_vectors_path, sigma_vectors_path, log_sigmas=log_sigmas)
    prefix = os.path.dirname(os.path.realpath(__file__))
    test_path = prefix + test_path
    total = 0
    correct = 0
    seen = 0
    total_cos = 0
    correct_log_var = 0

    if vocab is not None:
        correct_fr = 0  # we wil compute how many correct answers are obtained via pure frequency

    with open(test_path) as f:
        if header:
            f.next()
        for line in f:
            # print(line.split(' '))
            f_word, s_word, entail = line.split(' ')
            entail = entail.strip()
            entail = 1 if entail == "True" else -1
            if f_word in mus_and_sigmas and s_word in mus_and_sigmas:
                seen += 1
                f_mu, f_sigma = mus_and_sigmas[f_word]
                s_mu, s_sigma = mus_and_sigmas[s_word]
                score_cor = KL(f_mu, f_sigma, s_mu, s_sigma)
                score_wrong = KL(s_mu, s_sigma, f_mu, f_sigma)
                score = score_wrong - score_cor
                sign = np.sign(score)
                cos = cosine_sim(f_mu, s_mu)
                total_cos+=cos
                mess = "%s => %s sign: %d cos_sim: %f, KLs dif. %f" % (f_word, s_word, sign, cos, score)
                if vocab is not None:
                    f_freq = vocab[f_word].count
                    s_freq = vocab[s_word].count
                    correct_fr += (np.sign(s_freq - f_freq) == entail)
                    mess = "%s(%d) => %s(%d) sign: %d cos_sim: %f, KLs dif. %f" % (f_word, f_freq, s_word,s_freq, sign, cos, score)
                if debug:
                    print mess
                # TODO: what about score_cor == score_wrong?
                correct += (sign == entail)
                correct_log_var += (np.sign(variance(s_sigma, "True") - variance(f_sigma, "True")) == entail)
            total += 1
    print " ---------------------------------------------------- "
    print " ------------- DIRECTIONAL ENTAILMENT TEST -------------"
    print "------------- %s -------------" % test_path
    print "accuracy: %f %%" % (float(correct)/float(seen) * 100)
    if vocab is not None:
        print "accuracy based on freq: %f %%" % (float(correct_fr)/float(seen) * 100)
    print "accuracy based on log_var: %f %%" % (float(correct_log_var)/float(seen) * 100)
    print "seen: %f %%" % (float(seen)/float(total) * 100)
    print "average cos_sim %f" % (total_cos/float(seen))
    print "total sub-tasks: %d"% total