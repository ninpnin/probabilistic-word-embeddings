import numpy as np
import tensorflow as tf
from .utils import dict_to_tf
import progressbar
import warnings

def filter_rare_words(data, limit=5, keep_words=set()):
    """
    Filter out words that only occur a handful of times
    
    Args:
        data (list): list of strs
        limit: (int): words with fewer occurences are discarded
        keep_words (set): set of words that are always included regardless of frequency
    
    Returns:
        data, counts: list of strs and word counts as a dict
    """
    counts = {}
    if isinstance(data[0], list):
        for dataset in progressbar.progressbar(data):
            for wd in dataset:
                counts[wd] = counts.get(wd, 0) + 1

        outdata = []
        for dataset in data:
            outdataset = [wd for wd in progressbar.progressbar(dataset) if counts[wd] >= limit or wd in keep_words]
            outdata.append(outdataset)
    else:
        for wd in progressbar.progressbar(data):
            counts[wd] = counts.get(wd, 0) + 1
            
        outdata = [wd for wd in progressbar.progressbar(data) if counts[wd] >= limit or wd in keep_words]
    counts = {wd: count for wd, count in counts.items() if count >= limit or wd in keep_words}
    return outdata, counts

def downsample_common_words(data, counts, cutoff=0.00001, chunk_len=5000000, seed=None):
    """
    Discard words with the probability of 1 - sqrt(10^-5 / freq)
    Initially proposed by Mikolov et al. (2013)
    
    Args:
        data (Union[list, tf.Tensor]): list of strs or tf.Tensor of strings
        counts (dict): word counts in the dataset
        cutoff (float): lowest word frequency for when downsampling is applied
        chunk_len (int): the number of words that are processed at a time to restrict memory use

    Returns:
        list of strs
    """
    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    print("Discard some instances of the most common words...")
    N = sum(counts.values())
    counts_tf = dict_to_tf(counts)
        # Randomize and fetch by this probability
    if seed is not None:
        tf.random.set_seed(seed)
        
    if len(data) < chunk_len:
        frequencies = counts_tf.lookup(data) / N
        # Discard probability based on relative frequency
        probs = 1. - tf.sqrt(cutoff / frequencies)

        rands = tf.random.uniform(shape=data.shape, dtype=tf.float64)
        kepts = rands > probs
        indices = tf.where(kepts)
        newdata = tf.gather_nd(data, indices)
        return [wd.decode("utf-8") for wd in newdata.numpy()]
    else:
        l = []
        for i in progressbar.progressbar(list(range(len(data) // chunk_len))):
            i_prime = i * chunk_len
            chunk = data[i_prime:i_prime + chunk_len]
            frequencies = counts_tf.lookup(chunk) / N
            probs = 1. - tf.sqrt(cutoff / frequencies)
            rands = tf.random.uniform(shape=chunk.shape, dtype=tf.float64)
            kepts = rands > probs
            indices = tf.where(kepts)
            newdata = tf.gather_nd(chunk, indices)
            l = l + [wd.decode("utf-8") for wd in newdata.numpy()]

        return l

def preprocess_standard(text, keep_words=set(), limit=5, downsample=True, seed=None):
    """
    Standard preprocessing: filter out rare (<=5 occurences) words, downsample common words.

    Args:
        text (list): text as a list of strs
        keep_words: (set): words that are kept in the vocabulary regardless of frequency
        limit: (int): words with fewer occurences are discarded
        downsample: whether to do downsampling of common words

    Returns:
        text, vocabulary: text as a list of strs, vocabulary as a set of strs
    """
    N = len(text)
    text, counts = filter_rare_words(text, limit=limit, keep_words=keep_words)
    if downsample:
        text = downsample_common_words(text, counts, seed=seed)

    vocabulary = set(text)
    freqs = {wd: counts[wd] / N for wd in list(vocabulary)}
    return text, freqs

def preprocess_partitioned(texts, labels=None, keep_words=set(), limit=5, downsample=True, seed=None):
    """
    Standard preprocessing for partitioned datasets: filter out rare (<=5 occurences) words, downsample common words.

    Args:
        texts (list): list of texts, each element of which is a list of strs
        labels (list): label associated with each 
        keep_words: (set): words that are kept in the vocabulary regardless of frequency
        limit: (int): words with fewer occurences are discarded
        downsample: whether to do downsampling of common words

    Returns:
        text, vocabulary: text as a list of list of strs, vocabulary as a set of strs
    """
    if labels is not None:
        assert len(texts) == len(labels), "Number of data partitions and labels must be equal"
    assert isinstance(texts[0], list), "Data should be provided as a list of lists"
    N = sum([len(t) for t in texts])
    texts, counts = filter_rare_words(texts, limit=limit, keep_words=keep_words)
    if downsample:
        texts = [downsample_common_words(text, counts, seed=seed) for text in texts]

    def add_subscript(t, subscript):
        if len(t) == 0:
            warnings.warn(f"Empty text encountered {t}")
            return t

        if not isinstance(t, tf.Tensor):
            t = tf.constant(t)
        try:
            t = t + f"_{subscript}"
        except Exception as e:
            print(f"Error encountered with t: {t}; subscript {subscript}")
            raise e

        t = [wd.decode("utf-8") for wd in t.numpy()]
        return t

    if labels is not None:
        texts = [add_subscript(text, label) for text, label in progressbar.progressbar(zip(texts, labels))]
    vocabs = [set(text) for text in progressbar.progressbar(texts)]
    empty = set()
    vocabulary = empty.union(*vocabs)


    def _remove_subscript(wd):
        s = wd.split("_")
        n = len(s)
        return "_".join(s[:n-1])

    if labels is None:
        unnormalized_freqs = {wd: counts[wd] / N for wd in list(vocabulary)}
    else:
        unnormalized_freqs = {wd: counts[_remove_subscript(wd)] / N for wd in list(vocabulary)}
    freqs_sum = sum(unnormalized_freqs.values())
    freqs = {wd: f / freqs_sum for wd, f in unnormalized_freqs.items()}
    return texts, freqs