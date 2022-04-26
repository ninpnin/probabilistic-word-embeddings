import numpy as np
import tensorflow as tf
from .utils import dict_to_tf, transitive_dict

def filter_rare_words(data, limit=5):
    """
    Filter out words that only occur a handful of times
    
    Args:
        data (list): list of strs
    
    Returns:
        data, counts: list of strs and word counts as a dict
    """
    counts = {}
    if isinstance(data[0], list):
        for dataset in data:
            for wd in dataset:
                counts[wd] = counts.get(wd, 0) + 1

        outdata = []
        for dataset in data:
            outdataset = [wd for wd in dataset if counts[wd] >= limit]
            outdata.append(outdataset)
    else:
        for wd in data:
            counts[wd] = counts.get(wd, 0) + 1
            
        outdata = [wd for wd in data if counts[wd] >= limit]
    counts = {wd: count for wd, count in counts.items() if count >= limit}
    return outdata, counts

def downsample_common_words(data, counts, cutoff=0.00001):
    """
    Discard words with the probability of 1 - sqrt(10^-5 / freq)
    Initially proposed by Mikolov et al. (2013)
    
    Args:
        data (Union[list, tf.Tensor]): list of strs or tf.Tensor of strings
        counts (dict): word counts in the dataset
        cutoff (float): lowest word frequency for when downsampling is applied
    
    Returns:
        list of strs
    """
    if not isinstance(data, tf.Tensor):
        data = tf.constant(data)

    print("Discard some instances of the most common words...")
    N = sum(counts.values())
    counts_tf = dict_to_tf(counts)
    
    frequencies = counts_tf.lookup(data) / N
    # Discard probability based on relative frequency
    probs = 1. - tf.sqrt(cutoff / frequencies)
    # Randomize and fetch by this probability
    rands = tf.random.uniform(shape=data.shape, dtype=tf.float64)
    kepts = rands > probs
    indices = tf.where(kepts)
    newdata = tf.gather_nd(data, indices)
    return [wd.decode("utf-8") for wd in newdata.numpy()]

def preprocess_standard(text):
    """
    Standard preprocessing: filter out rare (<=5 occurences) words, downsample common words.

    Args:
        text (list): text as a list of strs

    Returns:
        text, vocabulary: text as a list of strs, vocabulary as a set of strs
    """
    text, counts = filter_rare_words(text)
    text = downsample_common_words(text, counts)

    vocabulary = set(text)
    return text, vocabulary

def preprocess_dynamic(texts):
    texts, counts = filter_rare_words(texts)
    #texts = [downsample_common_words(text, counts) for text in texts]

    vocabulary = set({wd for wd, count in counts.items() if count >= 5})
    return texts, vocabulary