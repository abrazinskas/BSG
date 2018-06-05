from libraries.data_iterators.open_text_data_iterator import OpenTextDataIterator
from libraries.tools.vocabulary import Vocabulary
from libraries.tokenizers.bsg_tokenizer import BSGTokenizer
from interfaces.i_bsg import IBSG
from libraries.misc.optimizations import Adam


class InterfaceConfigurator:
    """
    A class for configuring the model's interface. One can alter hyper-params in get_interface.

    """
    def __init__(self):
        pass

    @staticmethod
    def get_interface(train_data_path, vocab_file_path, output_folder_path=None, params_file_path=None, model_file_path=None):

        # Hyper-parameters
        half_window_size = 5  # (one sided)
        input_dim = 100
        h_dim = 100  # the number of components in the first hidden layers
        z_dim = 100  # the number of dimensions of the latent vectors
        alpha = 0.0075  # learning rate
        subsampling_threshold = None
        nr_neg_samples = 10
        margin = 5.0  # margin in the hinge loss
        epochs = 1
        max_vocab_size = 10000
        batch_size = 500

        tokenizer = BSGTokenizer(word_processor_type='open_text', use_external_tokenizer=False)
        data_iterator = OpenTextDataIterator(tokenizer=tokenizer)

        vocab = Vocabulary(data_iterator, max_size=max_vocab_size, min_count=1)
        vocab.load_or_create(vocab_file_path, train_data_path)
        vocab.assign_distr()

        lr_opt = Adam(learning_rate=alpha, beta1=0.9, beta2=0.999)

        i_model = IBSG(vocab=vocab, data_iterator=data_iterator, train_data_path=train_data_path, epochs=epochs,
                       half_window_size=half_window_size, nr_neg_samples=nr_neg_samples, subsampling_threshold=subsampling_threshold,
                       batch_size=batch_size, output_dir=output_folder_path)

        if model_file_path:
            i_model.load_model(model_file_path)
        else:
            i_model.init_model(vocab_size=len(vocab), input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, lr_opt=lr_opt, margin=margin)

        # load params only if the model was not loaded already
        if params_file_path and not model_file_path:
            i_model.load_params(params_file_path)

        return i_model

