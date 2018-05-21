# -*- coding: utf-8 -*-
# contains helper functions that are used in different iterators
import numpy as np
np.random.seed(1)

def pad_sents(sentences, max_length, pad_symbol, mask_current=False, padding_mode='both'):
    """
    pads many sentences
    :param mask_current: whether sentence elements that have pad symbols should be masked
    :param padding_mode: whether pad only the left side(dead useful for LSTM)

    """
    padded_sentences, masks = [], []
    for sentence in sentences:
        x, m = pad_sent(sentence, max_length, pad_symbol, mask_current=mask_current,
                        padding_mode=padding_mode)
        padded_sentences.append(x)
        masks.append(m)
    return np.array(padded_sentences, dtype="int32"), np.array(masks, dtype="float32")


# pads sentence if necessary with some symbol's id
# note that it returns a binary mask too
# TODO: write a better documentation for this function!
def pad_sent(sentence, max_length, pad_symbol, mask_current=False, padding_mode='both'):
    assert padding_mode in ['left', 'both', 'right']
    pad_number = max_length - len(sentence)

    if mask_current:
        masked_sentence = [s != pad_symbol for s in sentence]

    # will perform truncation if the sentence is too long
    if pad_number < 0:
        if mask_current:
            mask = masked_sentence
        else:
            mask = np.ones((max_length, ))
        res = sentence[:max_length]
        return res, mask

    # padding only left side
    if padding_mode == 'left':
        res = [pad_symbol] * pad_number + sentence
        if mask_current:
            mask = [0]*pad_number + masked_sentence
        else:
            mask = [0]*pad_number + [1]*len(sentence)
    # padding both sides
    elif padding_mode == 'both':
        res = [pad_symbol] * (pad_number/2) + sentence + [pad_symbol] * (pad_number/2)
        if mask_current:
            mask = [0]*(pad_number/2) + masked_sentence + [0]*(pad_number/2)
        else:
            mask = [0]*(pad_number/2) + [1]*len(sentence) + [0]*(pad_number/2)
        if pad_number % 2 == 1:
            res += [pad_symbol]
            mask += [0]
    # pad only the right side
    elif padding_mode == "right":
        res = sentence + [pad_symbol] * pad_number
        if mask_current:
            mask = masked_sentence + [0]*pad_number
        else:
            mask = [1]*len(sentence) + [0]*pad_number

    return res, mask


def allow_with_prob(word_count, total_words_count, subsampling_threshold=1e-5):
    """
    Sub-sampling of frequent words: can improve both accuracy and speed for large data sets
    Source: "Distributed Representations of Words and Phrases and their Compositionality".

    """
    freq = float(word_count) / float(total_words_count)
    removal_prob = 1.0 - np.sqrt(subsampling_threshold / freq)
    return np.random.random_sample() > removal_prob


def create_context_windows(sentence, half_window_size=0):
    """
    A generic function that either returns (center_word, window_context) or (center_word, left_context, right_context).
    To switch from first to the second mode, set half_window_size=0

    """
    n = len(sentence)
    for idx in range(half_window_size, n - half_window_size):
        center_word = sentence[idx]
        if half_window_size > 0:
            context = sentence[idx - half_window_size:idx] + sentence[idx + 1:idx + half_window_size + 1]
            yield (center_word, context)
        else:
            assert half_window_size == 0
            yield (center_word, sentence[0:idx], sentence[idx + 1:])


def create_continues_context_windows(sentence, special_center_words, half_window_size):
    """
    Creates windows where center words are separately marked(by using differnt vocab_ids) from context words.
    This function was used for the BSG's LSTM data batcher.

    """
    n = len(sentence)
    for idx in range(half_window_size, n - half_window_size):
        context = sentence[idx - half_window_size:idx] + sentence[idx + 1:idx + half_window_size + 1]
        context_and_center = sentence[idx - half_window_size:idx] + [special_center_words[idx]] + sentence[idx + 1:idx + half_window_size + 1]
        yield (sentence[idx], context, context_and_center)


def sample_words(batch_size,  vocab_size, distr, nr_neg_samples=1):
    return np.random.choice(size=(batch_size, nr_neg_samples), replace=True, a=vocab_size, p=distr)