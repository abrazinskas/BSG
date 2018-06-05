import theano
from theano import tensor as T, printing
from bword2vec import BWord2Vec
from layers.custom.bsg_encoder import BSGEncoder
from layers.standard.dense import Dense
from layers.standard.embeddings import Embeddings
from libraries.utils.other import merge_ordered_dicts


class BSG(BWord2Vec):
    """
    Theano implementation of the Bayesian Skip-gram model.

    """
    def __init__(self, vocab_size, input_dim=50, hidden_dim=50, latent_dim=100, lr_opt=None, margin=1., model_name='BSG with the hinge loss'):
        """
        :param vocab_size: the number of unique words
        :param input_dim: the number of components in the encoder's word embeddings
        :param hidden_dim: the number of components in the encoder's hidden layer
        :param latent_dim: the number of components in the latent vector(also output word mu's)
        :param lr_opt: learning rate optimizer object (e.g. Adam)
        :param margin: margin constant present in the hinge loss

        """
        assert lr_opt is not None
        BWord2Vec.__init__(self)

        self.model_name = model_name
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr_opt = lr_opt
        self.learning_rate = lr_opt.alpha
        self.margin = margin

        # assign full parameters
        self.params_full = self.__build_model()

        # extract only the actual parameter data-structures(tensors) as those will be optimized
        self.params = [param.value for param in self.params_full.values()]

        # user accessible functions build ( e.g. training functions)
        self.__build_functions()

    def __compute_cost_components(self, pos_context_words, neg_context_words, center_words, mask):
        """
        Computes two main components that comprise the objective : maximum margin(hinge loss) and kl involving center words.
        :param pos_context_words: a tensor with context words ids [batch_size x window_size]
        :param neg_context_words: a tensor with context words ids [batch_size x window_size]
        :param center_words: a tensor with center words ids [batch_size x 1]
        :param mask: a tensor binary mask where 0 indicates a padding
        :return: margin [batch_size x 1], kl [batch_size x 1]

        """
        mu_q, sigma_q = self.__encode(context_words=pos_context_words, center_words=center_words, mask=mask)
        mu_p, sigma_p = self.__compute_prior_params(center_words)
        kl = self.kl(mu_q, sigma_q, mu_p, sigma_p)
        margin = self.__max_margin(mu_q=mu_q, sigma_q=sigma_q, pos_context_words=pos_context_words,
                                   neg_context_words=neg_context_words, mask=mask)

        return margin, kl

    def __max_margin(self, mu_q, sigma_q, pos_context_words, neg_context_words, mask):
        """
        Computes a sum over context words margin(hinge loss).
        :param pos_context_words:  a tensor with true context words ids [batch_size x window_size]
        :param neg_context_words: a tensor with negative context words ids [batch_size x window_size]
        :param mask: a tensor binary mask where 0 indicates a padding
        :return: tensor [batch_size x 1]

        """
        b, window_size = pos_context_words.shape
        sigma_q = T.repeat(sigma_q, window_size, axis=0)
        mu_q = T.repeat(mu_q, window_size, axis=0)

        pos_c_resh = pos_context_words.reshape((-1, ))
        mu_p_pos, sigma_p_pos = self.__compute_prior_params(pos_c_resh)

        neg_c_resh = neg_context_words.reshape((-1, ))
        mu_p_neg, sigma_p_neg = self.__compute_prior_params(neg_c_resh)

        kl_pos = self.kl(mu_q, sigma_q, mu_p_pos, sigma_p_pos).reshape((b, -1))
        kl_neg = self.kl(mu_q, sigma_q, mu_p_neg, sigma_p_neg).reshape((b, -1))

        # hard margin
        return T.sum(T.maximum(0.0, self.margin - kl_neg + kl_pos) * mask, axis=1)

    def __encode(self, context_words, center_words, mask):
        """
        Encodes center and context words considering the binary mask, and returns Gaussian parameters.
        :param center_words: a tensor with center words ids
        :param context_words: a tensor with context words ids
        :param mask: a tensor binary mask where 0 indicates a padding.
        :return: mu [batch_size x latent_dim], sigma [batch_size x sigma_dim]

        """
        hidden = self.encoder(context_words, center_words, mask)

        # 2. perform affine transformations to generate Gaussian parameters
        return self.dense_mu(hidden), T.exp(self.dense_sigma(hidden))

    def __compute_prior_params(self, w):
        """
        :param w: a tensor with word ids
        :return: mean and sigma representations

        """
        return self.embeddings_mu(w), T.exp(self.embeddings_log_sigma(w, perform_dimshuffle=False))

    def __compute_log_determinant(self, w):
        log_sigma = T.log(self.__compute_prior_params(w)[1])
        return T.sum(log_sigma, axis=1)

    def __build_model(self):
        """
        Creates the actual model, returns parameters in the form of a dictionary.

        """
        var_dim = 1

        # the output representations of words(used in KL regularization and max_margin).
        self.embeddings_mu = Embeddings("emb_output_mu", self.vocab_size, self.latent_dim, init_type='uniform')
        self.embeddings_log_sigma = Embeddings("emb_output_log_sigma", self.vocab_size, var_dim, init_type='bsg_log_sigmas')

        # encoder corresponding layers
        self.encoder = BSGEncoder("encoder", input_dim=self.input_dim, output_dim=self.hidden_dim, collection_size=self.vocab_size)
        self.dense_mu = Dense("dense_mu", self.hidden_dim, self.latent_dim, init_type='xavier_uniform')
        self.dense_sigma = Dense("dense_sigma", self.hidden_dim, var_dim, init_type='xavier_uniform')

        # store all params into a variable
        full_params = merge_ordered_dicts(self.encoder.params, self.dense_mu.params, self.dense_sigma.params,
                                          self.embeddings_mu.params, self.embeddings_log_sigma.params)
        return full_params

    def __build_functions(self):
        """
        a general build function for bayesian word2vec models, it compiles main functions, such as train() and
        compute_loss()

        """
        pos_context_words = T.imatrix()  # context words (batch)
        neg_context_words = T.imatrix()
        center_words = T.ivector()  # center words (batch)
        mask = T.matrix()  # binary mask

        margin, kl = self.__compute_cost_components(pos_context_words, neg_context_words, center_words, mask)
        mean_kl = T.mean(kl, axis=0)
        mean_margin = T.mean(margin, axis=0)
        cost = mean_margin + mean_kl

        updates = self.lr_opt(cost, self.params)
        avg_log_det = T.mean(self.__compute_log_determinant(center_words))

        self.train = theano.function(inputs=[pos_context_words, neg_context_words, center_words, mask],
                                     outputs=[mean_margin, mean_kl, avg_log_det], updates=updates, on_unused_input='warn')
        self.compute_loss = theano.function(inputs=[pos_context_words, neg_context_words, center_words, mask],
                                            outputs=[margin, kl, avg_log_det], on_unused_input='warn')

        # representations extraction functions
        word_idx = T.iscalar()
        mu, sigma = self.__compute_prior_params(word_idx)
        self.get_word_mu_rep = theano.function(inputs=[word_idx], outputs=mu)
        self.get_word_sigma_rep = theano.function(inputs=[word_idx], outputs=sigma)

        # indicates what functions should be used for embeddings extraction
        self.repr_types = {
            "mu": self.get_word_mu_rep,
            "sigma": self.get_word_sigma_rep,
            }

        # the below function will be used in lexical substitution and other experiments
        self.encode = theano.function(inputs=[pos_context_words, center_words, mask],
                                      outputs=self.__encode(center_words=center_words, context_words=pos_context_words,
                                                            mask=mask))