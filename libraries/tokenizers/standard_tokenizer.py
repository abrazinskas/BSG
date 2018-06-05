from libraries.tools.word_processor import WordProcessor
from nltk import word_tokenize as external_tokenizer


# A more advanced tokenizer that both tokenizes and cleans textual data
class StandardTokenizer():

    def __init__(self, word_processor_type='none', use_external_tokenizer=True):
        """
        :param use_external_tokenizer: whether to use or word_tokenize tokenizer or rely on the simple splitter(x.split()).
        """
        if use_external_tokenizer:
            self.tokenizer = external_tokenizer
        else:
            self.tokenizer = lambda x: x.split()  # assuming that data was already tokenized
        self.word_processor = WordProcessor(word_processor_type=word_processor_type)

    def __call__(self, sentence):
        """
        :param sentence: a string of words
        :return: a list of clean tokens

        """
        words = self.tokenizer(sentence)
        tokens = []
        for word in words:
            token = self.word_processor(word)
            if token == "":
                continue
            tokens.append(token)
        return tokens