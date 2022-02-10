import numpy as np
import tensorflow as tf
from .utils import dict_to_tf, transitive_dict

def filter_rare_words(data, limit=5):
    counts = {}
    for wd in data:
        counts[wd] = counts.get(wd, 0) + 1
        
    outdata = [wd for wd in data if counts[wd] >= limit]
    return outdata, counts

# Discard words with the probability of 1 - sqrt(10^-5 / freq)
# Initially proposed by Mikolov et al. (2013)
#@tf.function
def downsample_common_words(data, counts, cutoff=0.00001):
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
    text, counts = filter_rare_words(text)
    text = downsample_common_words(text, counts)

    vocabulary = set(text)
    return text, vocabulary