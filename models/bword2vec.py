import pickle
import os
import theano
from support import load, write_vectors, kl_spher
from pickle import UnpicklingError
from libraries.tools.ordered_attrs import OrderedAttrs

## theano configuration
theano.optimizer_including = 'cudnn'


class BWord2Vec(OrderedAttrs):
    """
    Base class for the Bayesian Skip-gram model, it contains methods that can be used for multiple variants of BSG.

    """
    def __init__(self):
        OrderedAttrs.__init__(self)
        # the following attributes will be initialized in a child object
        self.params = None
        self.params_full = None
        self.repr_types = None

    @staticmethod
    def kl(mu_q, sigma_q, mu_p, sigma_p):
        """
        The generic Kullback Leibler function that passes arguments to the correct function

        """
        return kl_spher(mu_q, sigma_q, mu_p, sigma_p)

    def save_word_vectors(self, index_to_word, vectors_folder):
        """
        Extracts word vectors from different parameters and saves them to a desired vectors_folder destination
        :param index_to_word:  an array of words from vocab object
        :param vectors_folder: a desired destination path where word vectors should be saved

        """
        for name, func in self.repr_types.items():
            write_vectors(index_to_word, os.path.join(vectors_folder, name+".vectors"), func)

    def save_params(self, output_dir, output_file_name='params.pkl'):
        """
        Saves parameters via pickle to the output_dir under the specified name.

        """
        f = open(os.path.join(output_dir, output_file_name), 'wb')
        for param_name, param in self.params_full.items():
            # get_value() is necessary because param will be a tensor
            pickle.dump([param_name, param['values'].get_value()], f)
        f.close()

    def load_params(self, file_path, exclude_params=[]):
        """
        Loads params from a pickle saved file. The format has to correspond to the one that is used in save_params()

        """
        f = open(file_path, 'rb')
        initialized_params = []
        while True:
            try:
                param_name, param = pickle.load(f)
                if param_name in exclude_params:
                    continue
                self.initialize_param(param_name, param)
                initialized_params.append(param_name)
            except (EOFError, UnpicklingError):
                break
        f.close()
        return initialized_params

    def initialize_param(self, param_name, param_value):
        """
        Initializes a parameter with the provided values
        :param param_value: a matrix(array) of parameters

        """
        current_params = self.params_full
        if param_name not in current_params:
            raise ValueError("Could not find the parameter by '%s' name" % param_name)
        current_params[param_name]['values'].set_value(param_value)