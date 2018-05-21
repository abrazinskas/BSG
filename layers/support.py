import theano.tensor as T
import inspect

def inverse_sigmoid(decay_rate, batch_nr):
    """
    Computes inverse sigmoid function's value which is used in schedules sampling.
    :param decay_rate: hyper-parameter that controls the decay.

    """
    return decay_rate / (decay_rate + T.exp(batch_nr / decay_rate))


def select_matching_args(func, arguments_dict):
    """
    :return a hash with matching the function's arguments dictionary.
    """
    func_args = inspect.getargspec(func)[0]
    matching_args = {}
    for func_arg in func_args:
        if func_arg in arguments_dict:
            matching_args[func_arg] = arguments_dict[func_arg]
    return matching_args