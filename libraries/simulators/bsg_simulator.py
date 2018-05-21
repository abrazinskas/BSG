from base_simulator import BaseSimulator
import numpy as np


class BsgSimulator(BaseSimulator):
    """
    This class will both work for classical BSG and BSG with LSTM encoder.

    """
    def __init__(self, **kwargs):
        BaseSimulator.__init__(self, **kwargs)

    def get_representation(self, word):
        word_id = self.vocab[word].id
        mu = self.model.get_word_mu_rep(word_id)
        sigma = self.model.get_word_sigma_rep(word_id)
        return mu, sigma

    def encode(self, center_word, context_words):
        """
        :param center_word: int
        :param context_words: vector of ints
        """
        # convert to vocab_ids
        center_word_id = np.int32(self.vocab[center_word].id)
        context_word_ids = np.array([obj.id for obj in self.vocab[context_words]], dtype="int32")
        # generate the mask of ones
        mask = np.ones([1, len(context_words)], dtype="float32")
        mu, sigma = self.model.encode([context_word_ids], [center_word_id],  mask)
        return mu[0], sigma[0]
