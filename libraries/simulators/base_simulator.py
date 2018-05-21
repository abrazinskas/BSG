from support import load


class BaseSimulator:
    def __init__(self, vocab, model_file_path):
        """
        The passed in the .pkl format model has to have "encode" and "compute_prior_params" methods.

        """
        self.vocab = vocab
        self.model = load(model_file_path)
        assert hasattr(self.model, 'encode')
