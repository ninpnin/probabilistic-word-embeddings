"""
Definitions of the prior distributions of the embeddings.
"""
import numpy as np
import tensorflow as tf
import random
import progressbar

class Embedding:
    def __init__(self, vocabulary, dimensionality, lambda0=1.0, shared_context_vectors=True):
        assert isinstance(vocabulary, set)
        keys = list(vocabulary)
        if shared_context_vectors:
            keys = keys + list(set([key + "_c" for key in keys]))
        else:
            keys = keys + list(set([key.split("_")[0] + "_c" for key in keys]))
        keys = sorted(keys)
        values = tf.range(len(keys))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.vocabulary = tf.lookup.StaticHashTable(init, default_value=-1)
        self.theta = tf.Variable(np.random.rand(len(keys), dimensionality) - 0.5, dtype=tf.float64)
        self.lambda0 = lambda0

    @tf.function
    def __getitem__(self, item):
        if type(item) == str or isinstance(item, list):
            item = tf.constant(item)
        ix = self.vocabulary.lookup(item)
        return tf.gather(self.theta, ix, axis=0)

    def __len__(self):
        return len(self.theta)

    def log_prob(self, batch_size, data_size):
        plain_loss = - tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0
        return (batch_size / data_size) * plain_loss
