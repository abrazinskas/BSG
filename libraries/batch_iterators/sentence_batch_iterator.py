from support import allow_with_prob, sample_words, create_context_windows, pad_sents
from base_batch_iterator import BaseBatchIterator
from libraries.data_iterators.open_text_data_iterator import OpenTextDataIterator
from window_batch_iterator import Batch
from libraries.tools.vocabulary import PAD_TOKEN, UNK_TOKEN
import numpy as np
try:
    import re2 as re
except ImportError:
    import re


class SentenceBatchIterator(BaseBatchIterator):
    """
    Specific for LSTM based BSG iterator over batches.

    """

    def __init__(self, vocab, data_path, data_iterator, subsampling_threshold=None, batch_size=50,
                 max_sentence_length=None):
        """
        :param data_path: a path to data, can be a folder or a file path.
        :param subsampling_threshold: used in computation of words removal probability. The smaller the threshold
                                      the larger is the removal probability. In the original paper it was 1e-5.
                                      If None is passed, the subsampling will not be applied.

        """
        assert all([symbol in vocab for symbol in [PAD_TOKEN, UNK_TOKEN]])
        assert isinstance(data_iterator, OpenTextDataIterator)

        self.vocab = vocab
        self.data_path = data_path
        self.subsampling_threshold = subsampling_threshold
        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length

        self.data_iterator = data_iterator
        self.data_iterator.set_data_path(data_path)

        BaseBatchIterator.__init__(self)

    def load_data_batches_to_queue(self, queue):
        """
        Loads batches sequentially to a queue.

        """
        # create data holders(containers)
        left_context_tokens = []
        right_context_tokens = []
        center_tokens = []
        containers_current_size = 0
        max_length = 0
        for sentence, in self.data_iterator:
            # apply subsampling
            if self.subsampling_threshold:
                sentence = [token for token in sentence if allow_with_prob(self.vocab[token].count,
                                                                           self.vocab.total_count,
                                                                           subsampling_threshold=self.subsampling_threshold)]
            # convert to word_ids
            sentence_ids = [obj.id for obj in self.vocab[sentence]]

            # trim the sentence
            if self.max_sentence_length:
                sentence_ids = sentence_ids[:self.max_sentence_length]

            for center_token, left_context, right_context in create_context_windows(sentence_ids, half_window_size=0):
                # add to the data holders
                center_tokens.append(center_token)
                left_context_tokens.append(left_context)
                right_context_tokens.append(right_context)
                containers_current_size += 1
                max_length = max(max_length, len(left_context), len(right_context))
                # return the chunk/batch when the container gets full
                if containers_current_size >= self.batch_size:
                    batch = self.__create_batch(center_words=center_tokens, left_context=left_context_tokens,
                                                right_context=right_context_tokens, max_length=max_length)
                    queue.put(batch)

                    # reset
                    left_context_tokens = []
                    right_context_tokens = []
                    center_tokens = []
                    containers_current_size = 0
                    max_length = 0

        # return what has been collected if iteration is finished
        if containers_current_size > 0:
            batch = self.__create_batch(center_words=center_tokens, left_context=left_context_tokens,
                                        right_context=right_context_tokens, max_length=max_length)
            queue.put(batch)
        queue.put(None)  # to indicate that loading is finished

    def __create_batch(self, center_words, left_context, right_context, max_length):

        left_context, left_mask = pad_sents(left_context, max_length, pad_symbol=self.vocab[PAD_TOKEN].id,
                                            padding_mode='left')
        right_context, right_mask = pad_sents(right_context, max_length, pad_symbol=self.vocab[PAD_TOKEN].id,
                                              padding_mode='right')

        context = np.concatenate((left_context, right_context), axis=1)
        mask = np.concatenate((left_mask, right_mask), axis=1)

        # generate negative samples
        neg_context = sample_words(context.shape[0], len(self.vocab), self.vocab.uni_distr,
                                   nr_neg_samples=context.shape[1])
        neg_context = np.array(neg_context, dtype="int32")

        batch = Batch(pos_context_words=context, neg_context_words=neg_context, center_words=center_words, mask=mask)
        return batch