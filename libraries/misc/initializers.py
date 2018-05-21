import numpy as np
np.random.seed(42)

class Initializers():
    def __init__(self):
        pass

    @staticmethod
    def init(size, init_type="uniform"):
        """
        :param init_type: uniform
        :return: initialized numpy matrix with float32:
        """
        assert init_type in ['uniform', 'xavier_normal', 'xavier_uniform', 'zeros', 'bsg_log_sigmas']
        fans = compute_fans(size)
        if init_type == 'zeros':
            return np.zeros(shape=size, dtype="float32")
        if init_type == 'uniform':
            return np.float32(np.random.uniform(low=-0.05, high=0.05, size=size))
        if init_type == 'xavier_normal':
            return np.float32(np.random.normal(0.0, 2./np.sum(fans), size=size))
        if init_type == 'xavier_uniform':
            lim = np.sqrt(6.0 / np.sum(fans))
            return np.float32(np.random.uniform(low=-lim, high=lim, size=size))
        if init_type == 'bsg_log_sigmas':
            return np.float32(np.random.uniform(low=-3.5, high=-1.5, size=size))


def compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.
    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out



