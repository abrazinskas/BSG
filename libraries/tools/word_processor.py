# word processors that use regular expressions to clean words and tokenization
try:
    import re2 as re
except ImportError:
    import re


class WordProcessor:
    def __init__(self, word_processor_type='default'):
        self.__allowed_types = ['none', 'default', 'open_text']
        # sanity checks for input
        assert word_processor_type in self.__allowed_types
        # assigning processing function
        if word_processor_type == 'none':
            self.__call__ = lambda x: x
        if word_processor_type == 'default':
            self.__call__ = lambda word: re.sub(r'[^\w_,.?@!$#\':\/\-()]|[,\'?@$#]{2,}', "", word)
        if word_processor_type == "open_text":
            self.__call__ = self.__open_text_cleaner

    @staticmethod
    def __open_text_cleaner(word):
        """
        Direct copy from the original BSG setup. The tokens matching logic was moved to bsg_tokenizer.py

        """
        word = re.sub(r'[^\w\'\-]|[\'\-\_]{2,}', "", word)
        if len(word) == 1:
            word = re.sub(r'[^\daiu]', '', word)
        return word