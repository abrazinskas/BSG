# this file contains functions that are useful for Theano models.
from theano import tensor as T

def expand_dims(x, dim=-1):
    """Add a 1-sized dimension at index "dim".
    """
    # TODO: `keras_shape` inference.
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return x.dimshuffle(pattern)


def add_one_dim(x, dim=-1):
    pattern = list(x.shape)
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 1)
    return T.reshape(x, pattern)

def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    # TODO: `keras_shape` inference.
    shape = list(x.shape)
    shape.pop(axis)
    return T.reshape(x, tuple(shape))