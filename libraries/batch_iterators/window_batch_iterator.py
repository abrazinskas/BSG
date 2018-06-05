from support import pad_sents, allow_with_prob, sample_words, create_context_windows
from base_batch_iterator import BaseBatchIterator
from libraries.tools.vocabulary import PAD_TOKEN, UNK_TOKEN
import numpy as np
try:
    import re2 as re
except ImportError:
    import re


class Batch:
    def __init__(self, pos_context_words, neg_context_words, center_words, mask):
        self.pos_context_words = pos_context_words
        self.neg_context_words = neg_context_words
        self.center_words = center_words
        self.mask = mask

    def __len__(self):
        return self.pos_context_words.shape[0]


class WindowBatchIterator(BaseBatchIterator):

    def __init__(self, vocab, data_path, data_iterator, half_window_size=5, nr_neg_samples=5,
                 subsampling_threshold=None, batch_size=50):
        """
        :param data_path: a path to data, can be a folder or a file path.
        :param subsampling_threshold: used in computation of words removal probability. The smaller the threshold
                                      the larger is the removal probability. In the original paper it was 1e-5.
                                      If None is passed, the subsampling will not be applied.

        """
        assert all([symbol in vocab for symbol in [PAD_TOKEN, UNK_TOKEN]])
        self.vocab = vocab
        self.data_path = data_path
        self.half_window_size = half_window_size
        self.nr_neg_samples = nr_neg_samples
        self.subsampling_threshold = subsampling_threshold
        self.batch_size = batch_size

        self.data_iterator = data_iterator
        self.data_iterator.set_data_path(data_path)

        BaseBatchIterator.__init__(self)

    def load_data_batches_to_queue(self, queue):
        """
        Loads batches sequentially to a queue.

        """
        # create data placeholders
        pos_context_words = []
        center_words = []
        batch_current_size = 0
        max_length = 0

        for sentence, in self.data_iterator:

                # apply subsampling
                if self.subsampling_threshold:
                    sentence = [token for token in sentence if allow_with_prob(self.vocab[token].count,
                                                                               self.vocab.total_count,
                                                                               subsampling_threshold=self.subsampling_threshold)]
                # convert to word_ids
                sentence_ids = [obj.id for obj in self.vocab[sentence]]

                # pad corners
                sentence_ids = [self.vocab[PAD_TOKEN].id] * self.half_window_size + sentence_ids + \
                               [self.vocab[PAD_TOKEN].id] * self.half_window_size

                # create windows
                for center_token, context_tokens in create_context_windows(sentence_ids, self.half_window_size):
                    # add to the data holders
                    center_words.append(center_token)
                    pos_context_words.append(context_tokens)
                    batch_current_size += 1
                    max_length = max(max_length, len(context_tokens))

                    # return the chunk when the container gets full
                    if batch_current_size >= self.batch_size:
                        # generate negative samples
                        neg_context_words = sample_words(batch_current_size, len(self.vocab), self.vocab.uni_distr,
                                                         nr_neg_samples=self.nr_neg_samples)
                        batch = self.__create_batch(pos_context_words=pos_context_words,
                                                    neg_context_words=neg_context_words,
                                                    center_words=center_words, max_length=max_length)
                        queue.put(batch)

                        # reset
                        pos_context_words = []
                        center_words = []
                        batch_current_size = 0
                        max_length = 0

        # return what has been collected if iteration is finished
        if batch_current_size > 0:
            # generate negative samples
            neg_context_words = sample_words(batch_current_size, len(self.vocab), self.vocab.uni_distr,
                                             nr_neg_samples=self.nr_neg_samples)
            batch = self.__create_batch(pos_context_words=pos_context_words,
                                        neg_context_words=neg_context_words,
                                        center_words=center_words, max_length=max_length)
            queue.put(batch)
        queue.put(None)  # to indicate that loading is finished

    def __create_batch(self, pos_context_words, neg_context_words, center_words, max_length):

        neg_context_words = np.array(neg_context_words, dtype="int32")
        pos_context_words, mask = pad_sents(pos_context_words, max_length=max_length,
                                            pad_symbol=self.vocab[PAD_TOKEN].id, mask_current=True)
        center_words = np.array(center_words, dtype="int32")

        # convert all to numpy arrays
        # neg_context_words = np.array(neg_context_words, dtype="int32")
        pos_context_words = np.array(pos_context_words, dtype="int32")

        batch = Batch(pos_context_words=pos_context_words, neg_context_words=neg_context_words,
                      center_words=center_words, mask=mask)
        return batch