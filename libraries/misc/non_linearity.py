import theano.tensor as T


class NonLinearity():
    def __init__(self, type="linear"):
        assert type in ['linear', 'relu', 'sigmoid', 'hard_sigmoid', 'tanh']
        self.type = type

    def __call__(self, x):
        if self.type == 'linear':
            return x
        if self.type == "relu":
            return T.nnet.relu(x)
        if self.type == 'sigmoid':
            return T.nnet.sigmoid(x)
        if self.type == 'hard_sigmoid':
            return T.nnet.hard_sigmoid(x)
        if self.type == 'tanh':
            return T.tanh(x)


