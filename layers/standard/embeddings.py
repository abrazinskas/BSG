from layers.layer import Layer
from libraries.theano_support.extra import expand_dims


class Embeddings(Layer):
    def __init__(self, name, collection_size, output_dim, **kwargs):
        Layer.__init__(self, name=name, **kwargs)
        self.collection_size = collection_size
        self.output_dim = output_dim
        self.W = self.add_param("W", shape=(collection_size, output_dim), init_type=self.init_type,
                                regularizable=self.regularizable)

    def __call__(self, x, mask=None, perform_dimshuffle=True):
        """
        :return tensor [batch_size, output_dim, sequence_length] or [batch_size, output_dim]

        """
        # x = Print("x")(x)
        res = self.W[x]
        if mask:
            mask = expand_dims(mask, 2)
            res = res * mask
        # we want to make sure that output_dims are rows, and words are columns
        if res.ndim == 3 and perform_dimshuffle:
            res = res.dimshuffle((0, 2, 1))
        return res