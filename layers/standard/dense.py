from theano import tensor as T
from layers.layer import Layer
from libraries.misc.non_linearity import NonLinearity


class Dense(Layer):
    def __init__(self, name, input_dim, output_dim, non_linearity='linear', init_type='uniform', regularizable=True):
        """
        A simple fully connected layer that performs affine transformations following by a non-linearity.

        """
        Layer.__init__(self, name)
        self.non_linearity = NonLinearity(type=non_linearity)
        self.W = self.add_param(name="W", shape=(input_dim, output_dim), init_type=init_type,
                                regularizable=regularizable)
        self.b = self.add_param(name="b", shape=(output_dim, ), init_type='zeros')

    def __call__(self, x):
        """
        :param x: tensor [batch_size, input_dim]
        :return: tensor [batch_size, output_dim]

        """
        output = T.dot(x, self.W) + self.b
        output = self.non_linearity(output)
        return output



