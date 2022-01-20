"""
Definitions of the prior distributions of the embeddings.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
import functools
from collections import Counter

import numpy as np
import tensorflow as tf
import random
import progressbar

class Embedding:
    """Custom list that returns None instead of IndexError"""
    def __init__(self, vocabulary, dimensionality, lambda0=1.0):
        assert isinstance(vocabulary, set)
        keys = sorted(list(vocabulary))
        values = tf.range(len(keys))
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.vocabulary = tf.lookup.StaticHashTable(init, default_value=-1)
        self.theta = np.random.rand(len(keys), dimensionality) - 0.5
        self.lambda0 = lambda0

    def __getitem__(self, item):
        if type(item) == str or isinstance(item, list):
            item = tf.constant(item)
        print("Item", item)
        ix = self.vocabulary.lookup(item)
        print("Ix", ix)
        return tf.gather(self.theta, ix, axis=0)

    def __len__(self):
        return len(self.theta)

    def log_prob(self):
        return tf.reduce_sum(tf.multiply(self.theta, self.theta)) * self.lambda0

class LaplacianEmbedding(Embedding):
    """
    Embedding of shape (vocab_size, dim). Each embedding dimension \\( \\theta_{:j} \\) is IID distributed

    $$ \\theta_{:j} \\sim \\mathcal{N}( \\mathbf{0}, \\lambda_0 \\mathbf{I} + \\lambda_1 \\mathbf{L} ) $$

    i.e. the entries in one dimension are normally distributed with an augmented Laplacian as the precision matrix.

    Args:
        vocab_size: Vocabulary size of the embedding.
        dim: Dimensionality of each word vector.
        laplacian: The Laplacian matrix that is used to create the precision matrix. Requires the tf.SparseMatrix format.

            If no Laplacian is provided, it falls back to a zero matrix.
        lambda0: The diagonal weighting of the precision matrix. Corresponds to standard deviation if the Laplacian is a zero matrix.
        lambda1: The off-diagonal weighting of the precision matrix.
    """