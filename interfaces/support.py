import pickle
import sys
from collections import OrderedDict
sys.setrecursionlimit(10000)


def save(obj, file_path):
    file = open(file_path, 'wb+')
    pickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_path):
    file = open(file_path, 'rb+')
    return pickle.load(file)


def metrics_to_str(metrics, prefix=""):
    return prefix + " " + ", ".join(["%s: %f" % (name, value) for name, value in metrics.items()])


def infer_attributes_to_log(model):
    """
    Automatically infers parameters/attributes that should be logged. They are either ints/floats or strings.

    """
    all_attr = model.__dict__
    attr_to_log = OrderedDict()
    for attr_name, attr_value in all_attr.items():
        if isinstance(attr_value, (int, str, float, list)):
            attr_to_log[attr_name] = attr_value
    return attr_to_log


def format_experimental_setup(setup):
    """
    A specific for experiments writing function, that formats them property and write to the log file.
    :param params: a hash of params
    """
    st = ""
    st += '---------------------------- \n'
    st += '---- EXPERIMENT\'S SETUP ---- \n'
    for param_name, param_value in setup.iteritems():
        st += param_name + ": " + str(param_value) + '\n'
    st += '--------------------------'
    return st


def compute_loss(iterator, loss_func):
    """
    Computes the average loss over the whole dataset that is loaded to the iterator.

    """
    total_loss = 0.
    batch_size = iterator.batch_size
    datapoints_count = 0.
    # TODO: rethink if it's necessary to do all those mathematical manipulations
    for counter, batch in enumerate(iterator, 1):
        total_loss += loss_func(batch=batch)
        datapoints_count += len(batch)
    # rescale back as the loss was averaged over nr. of datapoints in each batch
    total_loss *= (batch_size / datapoints_count)
    return total_loss