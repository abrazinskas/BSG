# this file contains common functions that are used by models
import pickle
import numpy as np
from theano import tensor as T
from libraries.utils.paths_and_files import create_folders_if_not_exist
from pickle import UnpicklingError
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RandomStreams

seed = 1
r_stream = RandomStreams(seed=seed)
r_gpu_stream = MRG_RandomStreams(seed=seed)


def sample_words(self, batch_size, nr_neg_samples):
    """
    a function that is used for negative sampling context words, draws sample based on unigram distribution
    :param batch_size:
    :return: a matrix of size [batch_size x nr_of_negative_samples]

    """
    return self.r_stream.choice(size=(batch_size, nr_neg_samples), replace=True,
                                a=self.vocab_size, p=self.uni_distr, dtype='int32')


def kl_diag(mu_q, sigma_q, mu_p, sigma_p, eps):
    """
    Kullback Leibler divergence between two diagonal Gaussians
    :return: tensor [batch_size x 1]

    """
    d = mu_q.shape[1]
    sigma_p_inv = T.pow(sigma_p + 1e-6, -1)
    tra = T.sum(sigma_p_inv * sigma_q, axis=1)
    quadr = T.sum(sigma_p_inv * ((mu_p - mu_q)**2), axis=1)
    log_det_p = T.sum(T.log(sigma_p), axis=1)
    log_det_q = T.sum(T.log(sigma_q + eps), axis=1)
    log_det = log_det_p - log_det_q
    return 0.5 * (tra + quadr - d + log_det)


def kl_spher(mu_q, sigma_q, mu_p, sigma_p):
    """
    Kullback Leibler divergence between two spherical Gaussians
    :return: tensor [batch_size x 1]

    """
    d = mu_q.shape[1]
    sigma_p_inv = (1.0/sigma_p)
    tra = d * sigma_q*sigma_p_inv
    quadr = sigma_p_inv * T.sum((mu_p - mu_q)**2, axis=1, keepdims=True)
    log_det = - d*T.log(sigma_q * sigma_p_inv)
    res = 0.5 * (tra + quadr - d + log_det)
    return res.reshape((-1, ))


def l2_sqrd(x, axis=1):
    return T.sum(x**2, axis=axis)


# uniform init so far only
def init_weights(size, low_high_factor=100, scale_factor=1.):
    """

    :param size: size of a matrix to initialize
    :param low_high_factor: a factor in the initialization (see code)
    :return: initialized matrix of the same size as "size"
    """
    return np.float32(scale_factor)*np.float32(np.random.uniform(low=-low_high_factor**-0.5, high=low_high_factor**-0.5, size=size))


def init_weights2(size, low_factor=-1, high_factor=1, scale_factor=1.):
    """
    similar to init_weights but with decoupled low and high factors
    :param size: size of a matrix to initialize
    :return: initialized matrix of the same size as "size"

    """
    return np.float32(scale_factor)*np.float32(np.random.uniform(low=low_factor, high=high_factor, size=size))


def write_vectors(vocab, file_path, embeddings_function):
    """
    # extracts word vectors via embeddings_function and writes them into a file
    :param vocab: vocabulary object
    :param file_path: where to write vectors
    :param embeddings_function: a function that takes word id as input and return a vector embedding
    """
    create_folders_if_not_exist(file_path)
    with open(file_path, 'w') as output_file:
        for word_obj in vocab:
            word_vec = embeddings_function(word_obj.id)
            output_file.write(word_obj.token + " " + " ".join(str(f) for f in word_vec)+"\n")


def load(file_path):
    """
    a parameters loading function that is used to pre-loading pre-trained parameters to a model
    :param file_path: a path of a file that contains parameters in the format [parm_name:parm]

    """
    f = open(file_path, 'rb')
    while True:
        try:
            name, param = pickle.load(f)
            yield name, param
        except (EOFError, UnpicklingError):
            break
    f.close()