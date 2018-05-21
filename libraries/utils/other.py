import operator
from collections import OrderedDict

# def create_special_symbols_hash(special_symbols):
#     new_special_symbols = {}
#     for ss in special_symbols:
#         new_special_symbols[ss] = "<"+ss+">"
#     return new_special_symbols


# for float comparison
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def sort_hash(hash, by_key=True, reverse=True):
    if by_key:
        indx = 0
    else:
        indx = 1
    return sorted(hash.items(), key=operator.itemgetter(indx), reverse=reverse)


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def merge_ordered_dicts(*args):
    """
    Assuming that each collection is an Ordered dictionary, merges them into one.

    """
    new_params_dict = OrderedDict()
    for params in args:
        assert isinstance(params, OrderedDict)
        for key, value in params.items():
            new_params_dict[key] = value
    return new_params_dict


def append_to_ordered_dict(initial_dict, param_dict):
    """
    Assuming that collection is an Ordered dictionary, appends parameters to the initial one.

    """
    assert isinstance(param_dict, OrderedDict)
    for key, value in param_dict.items():
        initial_dict[key] = value
    return initial_dict