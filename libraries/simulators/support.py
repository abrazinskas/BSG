# functions specific for simulation
import numpy as np
import operator
import pickle
import sys
sys.setrecursionlimit(10000)


def load(file_path):
    return pickle.load(open(file_path, 'rb+'))


# returns KL divergence between two Gaussians or vMF
# KL(q||p)
# assumes [batch_size x z_dimension ] or [latent_dim, ] inputs
def KL(mu_q, sigma_q, mu_p, sigma_p, kl_type="gauss", debug=False):

    # adjusting dimensions
    flag = False
    if len(mu_q.shape) == 1 and len(sigma_q.shape) == 1 and len(mu_p.shape) == 1 and len(sigma_p.shape) == 1:
        mu_q = mu_q.reshape((1, -1))
        sigma_q = sigma_q.reshape((1, -1))
        mu_p = mu_p.reshape((1, -1))
        sigma_p = sigma_p.reshape((1, -1))
        flag = True

    if kl_type == "gauss":
        kl = KL_gauss(mu_q, sigma_q, mu_p, sigma_p, debug=debug)
    if kl_type == "vMF":
        kl = KL_vMF(mu_q, sigma_q, mu_p, sigma_p, debug=debug)
    if flag:
        kl = kl[0]
    return kl


# standard KL for Gaussians
def KL_gauss(mu_q, sigma_q, mu_p, sigma_p, debug=False):
    if sigma_q.shape[1] == 1 and sigma_p.shape[1] == 1:
        return KL_gauss_spherical(mu_q, sigma_q, mu_p, sigma_p, debug=debug)
    else:
        return KL_gauss_diagonal(mu_q, sigma_q, mu_p, sigma_p, debug=debug)

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


def KL_gauss_diagonal(mu_q, sigma_q, mu_p, sigma_p, debug=False, eps=1e-8):
    k = mu_q.shape[1]
    sigma_p_inv = 1./(sigma_p + eps)
    trace = np.sum(sigma_p_inv * sigma_q, axis=1)
    quadr = np.sum(sigma_p_inv * ((mu_p - mu_q)**2), axis=1)

    log_det_p = np.sum(np.log(sigma_p + eps), axis=1)
    log_det_q = np.sum(np.log(sigma_q + eps), axis=1)

    log_det = log_det_p - log_det_q

    if debug:
        print "trace : %f" % trace
        print "quadr : %f" % quadr
        print 'log_det_p : %f' % log_det_p
        print 'log_det_q : %f' % log_det_q
        print "log_det : %f" % log_det

    return 0.5 * (trace + quadr - k + log_det)


# KL for vMF
def KL_vMF(mu1, kappa1, mu2, kappa2, debug=False):
    return kl_vMF(mu1, kappa1, mu2, kappa2)

def pad_window_size(X, desired_window_size):
    current_height = X.shape[0]
    if desired_window_size <= current_height:
        return X
    pad_height = (desired_window_size- current_height)/2
    if current_height % 2 == 1:
        return np.pad(X, [(pad_height, pad_height+1), (0, 0)], 'constant')
    else:
        return np.pad(X, [(pad_height, pad_height), (0,0)], 'constant')




def sigmoid(x):
    return 1./(1 + np.exp(-x))

def simple_pmf(x):
    return 1./(1. + x)

def relu(x):
    return np.maximum(0, x)

def l2(x, axis=0):
    return np.sqrt(np.sum(x**2, axis=axis))

def cosine_sim(x, y):
    return float(np.sum(x*y))/float(np.sqrt(np.sum(x**2)*np.sum(y**2)))

# search_position: over what position to search KL(position 1 || ... ) or KL( ... || position 2)
def argmin_score(mu_q, sigma_q, mus_and_sigmas, num=1, type="kl", search_position=1):
        assert type in ["kl", "l2"]
        scores = {}
        for word, (mu_p, sigma_p) in mus_and_sigmas.items():
            if type == "kl":
                if search_position==1:
                    scores[word] = KL(mu_p, sigma_p, mu_q, sigma_q)
                else:
                    scores[word] = KL(mu_q, sigma_q, mu_p, sigma_p)

            else:
                scores[word] = l2(mu_q - mu_p, axis=0)
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
        return sorted_scores[0:num]

# search_position: over what position to search KL(position 1 || ... ) or KL( ... || position 2)
def closest_score(score, mu_q, sigma_q, mus_and_sigmas, num=1, type="kl", search_position=1):
    assert type in ["kl"]
    assert search_position in [1, 2]
    scores = {}

    for word, (mu_p, sigma_p) in mus_and_sigmas.items():
        if type == "kl":
            if search_position == 1:
                scores[word] = abs(score - KL(mu_p, sigma_p, mu_q, sigma_q))
            else:
                scores[word] = abs(score - KL(mu_q, sigma_q, mu_p, sigma_p))

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    return sorted_scores[0:num]

