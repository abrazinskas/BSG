import argparse
import numpy as np
import os


def glove_evaluate(vocab_file, vectors_file, bins=None, max_count=None):
    counts = {}
    words = []
    print("---------------------------------------------------------")
    print 'Reading %s' % vectors_file
    if max_count is not None:
        print "maximum frequency is %d (more freq. words are discarded)" % max_count
    with open(vocab_file, 'r') as f:
        # words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        for line in f:
            word, count = line.split(' ')
            count = int(count)
            if max_count is not None and max_count < count:
                continue
            counts[word] = count
            words.append(word)
    vocab = {w: idx for idx, w in enumerate(words)}
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                vectors[vals[0]] = [float(x) for x in vals[1:]]
    if bins is not None:
        hist, bin_edges = np.histogram(counts.values(), bins=bins)

    vocab_size = len(vocab)
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    # print "total vocab size is %d" % len(vocab)


    if bins is None:
        evaluate_vectors(W_norm, vocab)
    else:
        for i in range(len(bin_edges)-1):

            if hist[i] == 0:
                continue
            temp_vocab = {}
            temp_vectors = np.zeros((hist[i], vector_dim))
            edge1 = bin_edges[i]
            edge2 = bin_edges[i+1]
            j = 0  # current idx
            # collecting words that are between two edges
            for word, idx in vocab.items():
                if counts[word] >= edge1 and counts[word] <= edge2:
                    temp_vocab[word] = j
                    temp_vectors[j, :]= W_norm[idx]
                    j += 1
            # run evaluation
            print("---------------------------------------------------------")
            print "bin frequency limits are [%f, %f]" % (edge1, edge2)
            print "temp vocab's size is %d " % len(temp_vectors)
            print "temp vectors size is %d" % len(temp_vocab)
            evaluate_vectors(temp_vectors, temp_vocab, short=True)


def evaluate_vectors(W, vocab, short=False):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
        #prefix = './eval/question-data/'
    prefix = os.path.dirname(os.path.realpath(__file__))+'/question-data'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        if len(indices)==0: continue
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))
            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)
        if not short:
            print("%s:" % filenames[i])
            print('ACCURACY TOP1: %.2f%% (%d/%d)' %
                (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    if count_sem != 0:
        print('Semantic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    if count_syn:
        print('Syntactic accuracy: %.2f%%  (%i/%i)' %
            (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    if count_tot != 0:
        print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))

