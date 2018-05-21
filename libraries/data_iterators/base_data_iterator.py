from nltk import word_tokenize as default_tokenizer


class BaseDataIterator():

    def __init__(self, tokenizer=None, input_encoding='utf-8'):
        """
        Base data iterator that contains the general set_data_path method

        """
        self.tokenizer = tokenizer if tokenizer else default_tokenizer
        self.data_path = None
        self.input_encoding = input_encoding

    def set_data_path(self, data_path):
        self.data_path = data_path
