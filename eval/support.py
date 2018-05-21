import numpy as np


def KL(mu_q, sigma_q, mu_p, sigma_p, debug=False):
    """
    Kullback Leibler divergence implementation in numpy. Assumes [batch_size x z_dimension ] or [latent_dim, ] inputs.

    """
    # adjusting dimensions
    flag = False
    if len(mu_q.shape) == 1 and len(sigma_q.shape) == 1 and len(mu_p.shape) == 1 and len(sigma_p.shape) == 1:
        mu_q = mu_q.reshape((1, -1))
        sigma_q = sigma_q.reshape((1, -1))
        mu_p = mu_p.reshape((1, -1))
        sigma_p = sigma_p.reshape((1, -1))
        flag = True

    kl = KL_gauss_spherical(mu_q, sigma_q, mu_p, sigma_p, debug=debug)
    if flag:
        kl = kl[0]
    return kl


def KL_gauss_spherical(mu_q, sigma_q, mu_p, sigma_p, debug=False, eps=1e-8):
    k = mu_q.shape[1]
    sigma_p_inv = 1./(sigma_p + eps)
    trace = k * sigma_p_inv * sigma_q
    quadr = sigma_p_inv*np.sum(((mu_p - mu_q)**2), axis=1)
    log_det_p = np.log(sigma_p + 1e-10)
    log_det_q = np.log(sigma_q + 1e-10)
    log_det = k*(log_det_p - log_det_q)
    res = 0.5 * (trace + quadr - k + log_det)

    if debug:
        print "trace : %s" % str(trace)
        print "quadr : %s" % str(quadr)
        print 'log_det_p : %s' % str(log_det_p)
        print 'log_det_q : %s' % str(log_det_q)
        print "log_det : %s" % str(log_det)
        print 'res : %s'% str(res)
    return res.reshape((-1, ))


def cosine_sim(x, y):
    return float(np.sum(x*y))/float(np.sqrt(np.sum(x**2)*np.sum(y**2)))


def read_vectors_to_dict(mus_file_path, sigmas_file_path, log_sigmas=False, vocab=None, header=False):
    dict = {}
    with open(mus_file_path) as f:
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
    with open(sigmas_file_path) as f:
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