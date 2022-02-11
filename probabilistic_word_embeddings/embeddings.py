"""
Definitions of the prior distributions of the embeddings.
"""
import numpy as np
import tensorflow as tf
import random
import progressbar
from .utils import dict_to_tf

class Embedding:
    def __init__(self, vocabulary, dimensionality, lambda0=1.0, shared_context_vectors=True):
        assert isinstance(vocabulary, set)
        keys = list(vocabulary)
        if not shared_context_vectors:
            keys = keys + list(set([key + "_c" for key in keys]))
            self.vocabulary = {wd: ix for ix, wd in enumerate(sorted(keys))}
        else:
            context_keys = {key + "_c": key.split("_")[0] + "_c" for key in keys}
            all_indices = {key: ix for ix, key in enumerate(list(keys) + list(set(context_keys.values())))}
            word_dict = {key: all_indices[key] for key in keys}
            context_dict = {key: all_indices[value] for key, value in context_keys.items()}
            self.vocabulary = {**word_dict, **context_dict}
            assert max(self.vocabulary.values()) + 1 == len(set(context_dict.values())) + len(keys)
        unique_parameters = len(set(self.vocabulary.values()))
        self.tf_vocabulary = dict_to_tf(self.vocabulary)
        self.theta = tf.Variable(np.random.rand(unique_parameters, dimensionality) - 0.5, dtype=tf.float64)
        self.lambda0 = lambda0

    @tf.function
    def __getitem__(self, item):
        if type(item) == str:
            return self.theta[self.vocabulary[item]]
        elif isinstance(item, list):
            item = tf.constant(item)
        ix = self.tf_vocabulary.lookup(item)
        return tf.gather(self.theta, ix, axis=0)
    
    def __contains__(self, key):
        if type(key) == str:
            return key in self.vocabulary
        ix = self.tf_vocabulary.lookup(key)
        return ix != -1

    def log_prob(self, batch_size, data_size):
        plain_loss = - tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0
        return (batch_size / data_size) * plain_loss
