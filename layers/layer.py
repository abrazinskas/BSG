from collections import OrderedDict

from libraries.misc.initializers import Initializers
import theano


class Parameter:
    def __init__(self, name, shape, regularizable=False, init_type='uniform'):
        self.name = name
        self.value = theano.shared(Initializers.init(shape, init_type), name)
        self.regularizable = regularizable


class Layer:
    def __init__(self, name, init_type='xavier_uniform', regularizable=False):
        """
        A basic layer parent class of all other layers.

        """
        assert name is not None
        self.name = name
        self.init_type = init_type
        self.regularizable = regularizable
        self.params = OrderedDict()

    def get_params_to_reg(self):
        """
        Helper function that returns parameters that are necessary to regularize.

        """
        container = OrderedDict()
        for name in self.params.keys():
            if self.params[name].regularizable:
                container[name] = self.params[name]
        return container

    # TODO: think if I need to pass reg, and init_type like that if they are set as attributes already.
    def add_param(self, name, shape, regularizable=False, init_type='uniform'):
        """
        :param name: parameter's name
        :param shape: shape of the parameter
        :param regularizable: True/False depending on whether you want it be regularized
        :param init_type: what initialization to perform

        """
        assert name is not None
        name = "_".join([self.name, name])
        param = Parameter(name, shape=shape, regularizable=regularizable, init_type=init_type)
        self.params[name] = param
        return param.value