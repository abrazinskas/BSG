import re
import numpy as np
from pickle import UnpicklingError
from scipy.stats import chi2
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from vMF_support.vMF_KL import kl_vMF # I've disabled it to avoid calling theano for no reason

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from libraries.simulators.support import KL, cosine_sim, l2

import os
import pickle


# computes KL value for each distr of the word pair in the file
# returns 1) array with 2 columns : score and ent( 1 or 0)?, 2) seen number 3) total number of pairs
# scores are normalized
def get_kl_scores(mus_and_sigmas, filename, ignore_words=None, kl_type="gauss", normalize=True):
    assert kl_type in ['gauss', 'vMF']
    total = 0
    seen = 0
    res = []
    words = []
    with open(filename) as f:
        for sentence in f:
            f_word, s_word, entail = re.split(r'\t+', sentence)
            entail = entail.strip()
            entail = 1 if entail == "True" else 0
            if f_word in mus_and_sigmas and s_word in mus_and_sigmas:
                # skip the words which are not of the interest
                if ignore_words is not None and f_word not in ignore_words:
                    continue
                seen += 1
                # compute scores for each observed pair
                f_mu, f_sigma = mus_and_sigmas[f_word]
                s_mu, s_sigma = mus_and_sigmas[s_word]
                score = KL(f_mu, f_sigma, s_mu, s_sigma, kl_type=kl_type)
                # score = KL(s_mu, s_sigma, f_mu, f_sigma, kl_type=kl_type) # this one is wrong
                res.append([score, entail])
                words.append([f_word, s_word])
            total += 1
        res = np.array(res)
        # normalize scores
        if normalize:
            norm = np.sum(res[:, 0])
            res[:, 0] = res[:, 0]/norm
    return res, seen, total, np.array(words)

# computes KL value for each distr of the word pair in the file
# returns 1) array with 2 columns : score and ent( 1 or 0)?, 2) seen number 3) total number of pairs
# scores are normalized
def get_kl_scores_input_output(mus_and_sigmas_input, mus_and_sigmas_output, filename, ignore_words=None, kl_type="gauss", normalize=True):
    assert kl_type in ['gauss', 'vMF']
    total = 0
    seen = 0
    res = []
    words = []
    with open(filename) as f:
        for sentence in f:
            f_word, s_word, entail = re.split(r'\t+', sentence)
            entail = entail.strip()
            entail = 1 if entail == "True" else 0
            if f_word in mus_and_sigmas_input and s_word in mus_and_sigmas_output:
                # skip the words which are not of the interest
                if ignore_words is not None and f_word not in ignore_words:
                    continue
                seen += 1
                # compute scores for each observed pair
                f_mu, f_sigma = mus_and_sigmas_input[f_word]
                s_mu, s_sigma = mus_and_sigmas_output[s_word]
                score = KL(f_mu, f_sigma, s_mu, s_sigma, kl_type=kl_type)
                # score = KL(s_mu, s_sigma, f_mu, f_sigma, kl_type=kl_type) # this one is wrong
                res.append([score, entail])
                words.append([f_word, s_word])
            total += 1
        res = np.array(res)
        # normalize scores
        norm = np.sum(res[:, 0])
        if normalize:
            res[:, 0] = res[:, 0]/norm
    return res, seen, total, np.array(words)

def get_l2_scores(mus_and_sigmas, filename, ignore_words=None, kl_type="gauss", normalize=True):
    assert kl_type in ['gauss', 'vMF']
    total = 0
    seen = 0
    res = []
    words = []
    with open(filename) as f:
        for sentence in f:
            f_word, s_word, entail = re.split(r'\t+', sentence)
            entail = entail.strip()
            entail = 1 if entail == "True" else 0
            if f_word in mus_and_sigmas and s_word in mus_and_sigmas:
                # skip the words which are not of the interest
                if ignore_words is not None and f_word not in ignore_words:
                    continue
                seen += 1
                # compute scores for each observed pair
                f_mu, f_sigma = mus_and_sigmas[f_word]
                s_mu, s_sigma = mus_and_sigmas[s_word]
                score = l2(f_mu - s_mu)**2
                # score = KL(s_mu, s_sigma, f_mu, f_sigma, kl_type=kl_type) # this one is wrong
                res.append([score, entail])
                words.append([f_word, s_word])
            total += 1
        res = np.array(res)
        # normalize scores
        if normalize:
            norm = np.sum(res[:, 0])
            res[:, 0] = res[:, 0]/norm
    return res, seen, total, np.array(words)


def get_cos_scores(mus_and_sigmas, filename, ignore_words=None):
    total = 0
    seen = 0
    res = []
    words = []
    with open(filename) as f:
        for sentence in f:
            f_word, s_word, entail = re.split(r'\t+', sentence)
            entail = entail.strip()
            entail = 1 if entail == "True" else 0
            if f_word in mus_and_sigmas and s_word in mus_and_sigmas:
                # skip the words which are not of the interest
                if ignore_words is not None and f_word not in ignore_words:
                    continue
                seen += 1
                # compute scores for each observed pair
                f_mu, _ = mus_and_sigmas[f_word]
                s_mu, _ = mus_and_sigmas[s_word]
                score = cosine_sim(f_mu, s_mu)
                res.append([score, entail])
                words.append([f_word, s_word])

            total += 1
        res = np.array(res)
    return res, seen, total, np.array(words)


def read_vectors_to_dict(mus_file, sigmas_file, log_sigmas=False, vocab=None, header=False):
    dict = {}
    with open(mus_file) as f:
            for i, sentence in enumerate(f):
                if header and i==0:
                    continue

                parts = sentence.strip().split(" ")
                word = parts[0]

                # filter words that are not in vocab
                if vocab is not None and word not in vocab.word_to_index:
                    continue

                mu = np.array(parts[1:], dtype="float32")
                # normalize it
                # mu = mu / (np.sum(mu**2)**0.5)
                dict[word] = [mu]
                # print len(dict)
    with open(sigmas_file) as f:
            for i, sentence in enumerate(f):
                if header and i==0:
                    continue

                parts = sentence.strip().split(" ")
                word = parts[0]

                # filter words that are not in vocab
                if vocab is not None and word not in vocab.word_to_index:
                    continue

                sigma = np.array(parts[1:], dtype="float32")
                if log_sigmas:
                    sigma = np.exp(sigma)
                dict[word].append(sigma)
    return dict

def read_objects(file):
    loaded_objects = {}
    f = open(file, "rb")
    while True:
        try:
            obj = pickle.load(f)
            loaded_objects[obj[0]] = np.array(obj[1], dtype="float32")
        except (EOFError, UnpicklingError):
            break
    f.close()
    return loaded_objects


def read_vectors_to_arrays(mus_file, sigmas_file , log_sigmas=True):
    vocab = {}
    mus = []
    with open(mus_file) as f:
            for sentence in f:
                parts = sentence.split(" ")
                word = parts[0]
                vocab[word] = len(vocab)
                mus.append(np.array(parts[1:], dtype="float32"))
    sigmas = []
    # assuming the same order
    with open(sigmas_file) as f:
            for sentence in f:
                parts = sentence.split(" ")
                # word = parts[0]
                sigma = np.array(parts[1:], dtype="float32")
                if log_sigmas:
                    sigma = np.exp(sigma)
                sigmas.append(sigma)
    return np.array(mus), np.array(sigmas), vocab

def get_max_indx(ar, num_of_max=1):
    ar_sorted = sorted(ar, reverse=True)
    return [np.where(ar == ar_sorted[i]) for i in range(num_of_max)]
    # return [ar.index(ar_sorted[i]) for i in range(num_of_max)]


def print_variances(words, mus_and_sigmas, log=False):
    for word in words:
        mu, sigma = mus_and_sigmas[word]
        print '----------------'
        print 'word : %s' % word
        var = variance(sigma,log)
        mes = "log_var:" if log else "var:"
        mes += " %f" % var
        print mes
        print '----------------'


def variance(sigma, log=False):
    if log:
        var = np.sum(np.log(sigma))
    else:
        var = np.prod(sigma)
    return var




def create_dir_ent_dataset(ent_file, output_file, sep=" "):
    prefix = os.path.dirname(os.path.realpath(__file__))
    ent_file = prefix + ent_file
    o_f = open(output_file, 'w+')
    # include header
    o_f.write(sep.join(["word1", "word2", "word1 => word2"])+"\n")
    with open(ent_file) as f:
        for sentence in f:
             f_word, s_word, entail = re.split(r'\t+', sentence.strip())
             # print entail == "False"
             if entail == "False":
                 continue
             o_f.write(sep.join([f_word, s_word, "True"])+"\n")
             o_f.write(sep.join([s_word, f_word, "False"])+"\n")
    o_f.close()
    print "wrote output to %s"% output_file


# currently works only for one word-context
def encode(context, word, params):
    H = np.mean(params["V"][context]) + params["V"][word]
    mu = np.dot(params["W"], H.T) + params["b2"]  #.reshape((-1,))
    log_sigma = np.dot(params["U"], H.T) + params["b3"]  #.reshape((-1,))
    return mu, log_sigma

# extracts contexts and converts everything into tokens
# def extract_contexts(vocab, text_file_path, center_words, context_window):
#     center_words = [vocab.get_id(word) for word in center_words]
#     contexts = {word:[] for word in center_words}
#     for sentence in tokenize_files(vocab, text_file_path):
#         for idx, w in enumerate(sentence):
#             if w in contexts:
#                 start = max(0, idx - context_window)
#                 end = idx + context_window + 1
#                 context = sentence[start:idx] + sentence[idx:end]
#                 contexts[w].append(context)
#     return contexts

# def write_text(sentences, file):
#     with open(file, "w"):
#         for sentence in sentences:
#


def decode_to_words(vocab, contexts):
    decoded = {vocab.get_word(w):[] for w in contexts.keys()}
    for center_word, word_contexts in contexts.items():
        center_word = vocab.get_word(center_word)
        print " ---------------------------"
        print "center word:  %s" % center_word
        print "context is : "
        for context in word_contexts:
            decoded_context = []
            for c in context:
                decoded_context.append(vocab.get_word(c))
            decoded[center_word].append(decoded_context)
            print " ".join(decoded_context)
        print " ---------------------------"
    return decoded