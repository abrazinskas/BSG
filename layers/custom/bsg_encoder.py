from layers.layer import Layer
from layers.standard.embeddings import Embeddings
import theano.tensor as T
from libraries.misc.non_linearity import NonLinearity
from libraries.utils.other import merge_ordered_dicts


class BSGEncoder(Layer):
    """
    Encoder that is specific to the original BSG version. It uses one input representation of words, and performs
    transformation of context and center word representations.

    """
    def __init__(self, name, input_dim, output_dim, collection_size, non_linearity='relu'):
        Layer.__init__(self, name=name,init_type='uniform', regularizable=True)
        self.non_linearity = NonLinearity(type=non_linearity)
        self.embeddings = Embeddings(name="emb_encoder", collection_size=collection_size, output_dim=input_dim)
        self.C = self.add_param(name="C", shape=(2*input_dim, output_dim), init_type=self.init_type,
                                regularizable=self.regularizable)
        # store additional params
        self.params = merge_ordered_dicts(self.embeddings.params, self.params)

    def __call__(self, context_words, center_words, mask=None):
        """
        :param context_words: tensor [batch_size, seq_length]
        :param center_words: tensor [batch_size]
        :param mask: tensor [batch_size, seq_length]
        :return: tensor [batch_size, output_dim]

        """
        b, full_window_size = context_words.shape

        # 0. get representations
        repr_center = T.repeat(self.embeddings(center_words).dimshuffle([0, 'x', 1]), full_window_size, axis=1) \
                      * mask.dimshuffle([0, 1, "x"])
        repr_context = self.embeddings(context_words, mask, perform_dimshuffle=False)

        # 1. combine representations
        repr_common = T.concatenate([repr_center, repr_context], axis=2)

        # 2. compute hidden layer by summing common representations
        hidden = T.sum(self.non_linearity(T.dot(repr_common, self.C)), axis=1)

        return hidden