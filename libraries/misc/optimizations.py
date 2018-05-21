# This file contains learning rate optimizations
import lasagne


class LROpt:
    def __init__(self, learning_rate):
        self.alpha = learning_rate


class Adam(LROpt):
    def __init__(self, learning_rate, beta1, beta2):
        LROpt.__init__(self, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

    def __call__(self, cost, params):
        return lasagne.updates.adam(cost, params, learning_rate=self.alpha, beta1=self.beta1, beta2=self.beta2)


class SGD(LROpt):
    def __init__(self, learning_rate):
        LROpt.__init__(self, learning_rate)

    def __call__(self, cost, params):
        return lasagne.updates.sgd(cost, params, learning_rate=self.alpha)


class AdaGrad(LROpt):
    def __init__(self, learning_rate, eps):
        LROpt.__init__(self, learning_rate)
        self.eps = eps

    def __call__(self, cost, params):
        return lasagne.updates.adagrad(cost, params, learning_rate=self.alpha, epsilon=self.eps)

